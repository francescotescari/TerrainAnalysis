from enum import Enum

import wifi
import ubiquiti as ubnt
import random
from antenna import Antenna

node_fixed_cost = 200


class AntennasExahustion(Exception):
    msg = "No more antennas"
    pass


class LinkUnfeasibilty(Exception):
    msg = "Link Unfeasibile"
    pass


class ChannelExahustion(Exception):
    msg = "No more channels"
    pass


class CostChoice(Enum):
    NOT_INTERESTED = (1, "grey")
    LEAF_NODE = (2, "green")
    SUPER_NODE = (3, "blue")

    def __str__(self):
        return self.name

    def color(self):
        return self.value[1]


class Node:
    def __init__(self, max_ant):
        self.antennas = []
        self.max_ant = max_ant
        self.free_channels = wifi.channels[:]

    def cost(self):
        cost = node_fixed_cost
        for a in self.antennas:
            cost += a.device['average_price']
        return cost

    def __repr__(self):
        return (str(self))

    def __str__(self):
        string = ""
        for a in self.antennas:
            string += str(a) + "<br>"
        return string

    def __len__(self):
        return len(self.antennas)

    def check_channel(self, channel):
        if channel not in self.free_channels:
            raise ChannelExahustion
        self.free_channels.remove(channel)

    def add_antenna(self, loss, orientation, device=None, channel=None):
        # If the device is not provided we must find the best one for this link
        if not device:
            src_device = ubnt.get_fastest_link_hardware(loss)[1]
        else:
            src_device = ubnt.get_fastest_link_hardware(loss, device[0])[1]
        if not channel:
            channel = self._pick_channel()
        else:
            self.check_channel(channel)
        if not src_device:
            raise LinkUnfeasibilty
        if (len(self.antennas) >= self.max_ant):
            raise AntennasExahustion
        ant = Antenna(src_device, orientation, channel)
        self.antennas.append(ant)
        return ant

    def get_best_dst_antenna(self, link):
        result = None
        # filter the antennas that are directed toward the src
        # and on the same channel
        visible_antennas = [ant for ant in self.antennas
                            if ant.check_node_vis(link_angles=link['dst_orient'])]

        # sort them by the bitrate and take the fastest one
        if visible_antennas:
            best_ant = max(visible_antennas,
                           key=lambda x: ubnt.get_fastest_link_hardware(link['loss'],
                                                                        target=x.ubnt_device[0])[0])
            result = best_ant
        else:
            result = self.add_antenna(loss=link['loss'], orientation=link['dst_orient'])
        return result

    def _pick_channel(self):
        try:
            channel = random.sample(self.free_channels, 1)[0]
            self.free_channels.remove(channel)
            return channel
        except ValueError:
            raise ChannelExahustion


class CostNode(Node):
    """Extension of node object with cost choice property"""
    BAR_HTML = '<div class="pbar"><div class="ni" style="width:{:.0f}px"></div><div class="ln" style="width:{:.0f}px"></div><div class="sn" style="width:{:.0f}px"></div></div>'

    @staticmethod
    def gateway_node(max_ant, building):
        node = CostNode(max_ant, None)
        node.building = building
        node.properties['Is GW'] = True
        node.gid = building.gid
        return node

    def __init__(self, max_ant, applied_model):
        if applied_model is not None:
            ths = applied_model.get_probabilities()
            p1 = ths[0]
            p2 = ths[1] - ths[0]
            p3 = 1 - ths[1]
            self.probs = (p1, p2, p3)
            self.properties = {} if applied_model.properties is None else applied_model.properties
            self.properties["Probabilities"] = "{:.2f}% - {:.2f}% - {:.2f}%".format(p1 * 100, p2 * 100, p3 * 100)
            self.cost_choice = random.choices((CostChoice.NOT_INTERESTED, CostChoice.LEAF_NODE, CostChoice.SUPER_NODE),
                                              self.probs)[0]
            if self.cost_choice is CostChoice.NOT_INTERESTED:
                max_ant = 0
            elif self.cost_choice is CostChoice.LEAF_NODE:
                max_ant = 1

            self.building = applied_model.building
            self.gid = self.building.gid
        else:
            self.properties = {}
            self.building = None
            self.cost_choice = None
            self.gid = None
        self.add_failed = False
        self.model = applied_model
        super().__init__(max_ant)

    def get_weight(self):
        return 0 if self.model is None else self.model.weight

    def props_str(self):
        res = "Node type: " + str(self.cost_choice) + "<br>"
        for key, value in self.properties.items():
            try:
                if key == 'Fail cause' and not self.add_failed:
                    continue
                res += str(key) + ": " + str(value) + "<br>"
            except Exception:
                pass
        return res.replace("'", '"')

    def get_color(self):
        if self.add_failed or self.cost_choice is None:
            return "yellow"
        return self.cost_choice.color()

    def set_fail(self, cause):
        print("Failed to link node {}: {}".format(self.building.gid, cause))
        self.properties['Fail cause'] = cause

    def prob_bars(self):
        try:
            p1, p2, p3 = self.probs
            return self.BAR_HTML.format(p1 * 200, p2 * 200, p3 * 200)
        except Exception:
            return ''
