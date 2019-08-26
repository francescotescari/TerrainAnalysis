import collections
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


class HarmfulLink(Exception):
    msg = "Link harmful"
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

    ant_height = 5

    @staticmethod
    def gateway_node(max_ant, building):
        return Node(building, max_ant, properties={'Gateway': 'true'})

    def __init__(self, building, max_ant, cost_choice=None, properties=None):
        if building is None:
            raise TypeError("None building in node is not allowed")
        self.antennas = []
        self.max_ant = max_ant
        self.free_channels = wifi.channels[:]
        self.building = building
        self.add_failed = False
        self.properties = properties if isinstance(properties, collections.Mapping) else {}
        self.properties['Building gid'] = building.gid
        self.gid = building.gid << 8
        self.cost_choice = cost_choice
        self.available_devices = ['AM-NanoStation5ACL'] if cost_choice is CostChoice.LEAF_NODE else None

    def cost(self):
        cost = node_fixed_cost
        for a in self.antennas:
            cost += a.device['average_price']
        return cost

    def __repr__(self):
        return str(self.gid)

    def __str__(self):
        return "Node %d" % self.gid

    def __len__(self):
        return len(self.antennas)

    def check_channel(self, channel):
        if channel not in self.free_channels:
            raise ChannelExahustion
        self.free_channels.remove(channel)

    def add_antenna(self, loss, orientation, device=None, channel=None, force_device=None):
        av_devs = self.available_devices if force_device is None else force_device
        # If the device is not provided we must find the best one for this link
        if not device:
            src_device = ubnt.get_fastest_link_hardware(loss, available_devices=av_devs)[1]
        else:
            src_device = ubnt.get_fastest_link_hardware(loss, device[0], available_devices=av_devs)[1]
        if not channel:
            channel = self._pick_channel()
        else:
            self.check_channel(channel)
        if not src_device:
            raise LinkUnfeasibilty
        if len(self.antennas) >= self.max_ant:
            raise AntennasExahustion
        ant = Antenna(src_device, orientation, channel)
        self.antennas.append(ant)
        return ant

    def get_best_dst_antenna(self, link, pref_channels=None, force_device=None):
        av_devs = self.available_devices if force_device is None else force_device
        # filter the antennas that are directed toward the src
        # and on the same channel
        visible_antennas = [ant for ant in self.antennas
                            if (pref_channels is None or (ant.channel in pref_channels)) and ant.check_node_vis(link_angles=link['dst_orient'])]

        # sort them by the bitrate and take the fastest one
        if visible_antennas:
            best_ant = max(visible_antennas,
                           key=lambda x: ubnt.get_fastest_link_hardware(link['loss'],
                                                                        target=x.ubnt_device[0])[0])
            result = best_ant
        else:
            try:
                pref_channels = pref_channels & set(self.free_channels)
                pref_channel = random.sample(pref_channels, 1)[0]
            except Exception:
                pref_channel = None
            result = self.add_antenna(loss=link['loss'], orientation=link['dst_orient'], force_device=av_devs, channel=pref_channel)
        return result

    def _pick_channel(self):
        try:
            channel = random.sample(self.free_channels, 1)[0]
            self.free_channels.remove(channel)
            return channel
        except ValueError:
            raise ChannelExahustion

    def props_str(self):
        if self.antennas:
            self.properties['Antennas'] = '<br>'.join([str(x) for x in self.antennas])
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
        try:
            if self.add_failed:
                return "yellow"
            return self.cost_choice.color()
        except Exception:
            return "black"

    def set_fail(self, cause):
        print("Failed to link node {}: {}".format(self.gid, cause))
        self.properties['Fail cause'] = cause

    def coord_height(self):
        ch = self.building.coord_height()
        ch['height'] = self.ant_height
        ch['node'] = self
        return ch

    def xy(self):
        return self.building.xy()

    def __eq__(self, other):
        return self.gid == other.gid

    def __hash__(self):
        return self.gid


class CostNode(Node):
    BAR_HTML = '<div class="pbar"><div class="ni" style="width:{:.0f}px"></div><div class="ln" style="width:{:.0f}px"></div><div class="sn" style="width:{:.0f}px"></div></div>'

    def __init__(self, max_ant, applied_model):
        if applied_model is None:
            raise TypeError("Cannot create cost node: applied model object is None")
        ths = applied_model.get_probabilities()
        cost_choice = random.choices((CostChoice.NOT_INTERESTED, CostChoice.LEAF_NODE, CostChoice.SUPER_NODE),
                                     cum_weights=(ths[0], ths[1], 1))[0]
        if cost_choice is CostChoice.NOT_INTERESTED:
            max_ant = 0
        elif cost_choice is CostChoice.LEAF_NODE:
            max_ant = 1
        self.model = applied_model
        self.ths = ths
        super().__init__(applied_model.building, max_ant, cost_choice=cost_choice, properties=applied_model.properties)
        self.properties['Probabilities'] = '{:.1f}% - {:.1f}% - {:.1f}%'.format(ths[0] * 100,
                                                                                (ths[1] - ths[0]) * 100,
                                                                                (1 - ths[1]) * 100)

    def prob_bars(self):
        if self.ths is None or len(self.ths) < 2:
            return ''
        p1 = self.ths[0]
        p2 = self.ths[1] - self.ths[0]
        p3 = 1 - self.ths[1]
        return self.BAR_HTML.format(p1 * 200, p2 * 200, p3 * 200)

    def props_str(self):
        return super().props_str() + self.prob_bars()

    def get_weight(self):
        return 0 if self.model is None else self.model.weight
