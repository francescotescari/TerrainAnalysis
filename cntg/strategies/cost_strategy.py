from multiprocessing import Pool
import random

import numpy

from cn_generator import CN_Generator, NotInterestedNode, NoMoreNodes
from cost_model import CostError
from misc import Susceptible_Buffer
import time
from antenna import Antenna
import code
from node import LinkUnfeasibilty, AntennasExahustion, ChannelExahustion, CostChoice


class CostStrategy(CN_Generator):

    def __init__(self, args, unk_args=None):
        self.sb = Susceptible_Buffer()
        super().__init__(args=args, unk_args=unk_args)
        self._post_init()

    def stop_condition(self):
        if self.n:
            return self.stop_condition_maxnodes() or self.stop_condition_minbw()
        return self.stop_condition_minbw()

    def get_newnode(self):
        return self.get_random_node()

    def restructure(self):
        return self.restructure_edgeeffect_mt()

    def add_links_leaf(self, node, visible_links):
        visible_links.sort(key=lambda x: x['loss'], reverse=True)
        src_ant = False
        link = False
        added = self.add_node(node)
        while visible_links:
            link = visible_links.pop()
            try:
                src_ant = self.add_link(link)
                break
            except LinkUnfeasibilty as e:
                # If the link is unfeasible I don't need to try on the followings
                print(e.msg)
                self.noloss_cache[node].add(link['dst'])
                node.set_fail(e.msg)
                break
            except (AntennasExahustion, ChannelExahustion) as e:
                # If the antennas/channel of dst are finished i can try with another node
                self.noloss_cache[node].add(link['dst'])
                node.set_fail(e.msg)
        if not src_ant or not link:
            self.remove_node(node, added)
            return False
        self.feasible_links.append(link)
        return True

    def add_links_super(self, node, visible_links):
        result = set()
        building = node.building
        # Value of the bw of the net before the update
        min_bw = self.net.compute_minimum_bandwidth()
        # Let's create a dict that associate each link to a new net object.
        metrics = self.pool.starmap(self.net.calc_metric,
                                    [(link, self.node_for(link['src'])) for link in visible_links])
        # Filter out unwanted links
        clean_metrics = []
        # destination_set = self.super_nodes if new_node.cost_choice == CostChoice.SUPER_NODE else self.leaf_nodes
        for m in metrics:
            assert building == m['link']['src']  # link source is the building, right?
            if m['min_bw'] == 0:
                print("First Node")
                # This is the first node we add so we have to ignore the metric and add it
                self.add_node(node)
                src_ant = self.add_link(m['link'])
                return True
            if not m['min_bw']:
                # If is none there was an exception (link unaddable) thus we add it to cache
                # or if a leaf is trying to connect to the gateway
                self.noloss_cache[building].add(m['link']['dst'])
            else:
                clean_metrics.append(m)
        # We want the link that maximizes the difference of the worse case
        if not clean_metrics:
            # All the links are unfeasible for some reasnon (strange)
            node.set_fail("Empty clean metrics")
            return False
        # Order the links for min_bw difference and then for abs(loss) because we order from smallest to biggest

        ordered_metrics = sorted(clean_metrics,
                                 key=lambda m: (min_bw[m['node']] - m['min_bw'] *
                                                m['link']['loss']),
                                 reverse=True)
        links = [m['link'] for m in ordered_metrics]
        link = links.pop()
        print("Choosen link", link)
        result.add(link['dst'])
        self.add_node(node)
        # Don't need to try since the unvalid link have been excluded by calc_metric()
        src_ant = self.add_link(link)
        # Add the remaining links if needed
        link_in_viewshed = [l for l in links
                            if src_ant.check_node_vis(l['src_orient'])]
        link_added = 0
        while link_in_viewshed and link_added < self.V:
            link = link_in_viewshed.pop()
            visible_links.remove(link)  # remove it from visible_links af
            try:
                self.add_link(link, reverse=True)
            except (LinkUnfeasibilty, AntennasExahustion, ChannelExahustion) as e:
                print(e.msg)
            else:
                result.add(link['dst'])
                link_added += 1
        self.feasible_links += visible_links
        return result

    def add_links(self, new_node):
        building = new_node.building
        available_buildings = list(self.super_nodes.values())
        is_leaf = new_node.cost_choice is CostChoice.LEAF_NODE
        if not is_leaf:
            available_buildings.append(self.gw_node.building)
        # returns all the potential links in LoS with the new node
        print("Leaf nodes: %d, Super nodes: %d" % (len(self.leaf_nodes), len(self.super_nodes)))
        print("testing node %r, against %d potential nodes,"
              "already tested against %d nodes" %
              (building, len(available_buildings) - len(self.noloss_cache[building]),
               len(self.noloss_cache[building])))
        print("Super nodes to attach to: ", self.super_nodes.keys())
        # time.sleep(0.1)
        visible_links = [link for link in self.check_connectivity(
            available_buildings, building) if link]
        if not visible_links:
            new_node.set_fail("No visible links")
            return False
        return self.add_links_leaf(new_node,
                                   visible_links) if is_leaf else self.add_links_super(
            new_node, visible_links)


