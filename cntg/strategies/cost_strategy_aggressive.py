import numpy

from cost_model import CostError
from node import CostChoice, CostNode
from strategies.cost_strategy import CostStrategy


class CostStrategyAggressive(CostStrategy):

    def add_links(self, new_node):
        res = super().add_links(new_node)
        if res:
            self.add_linkable_nodes([new_node.building])
        return res

    def add_linkable_nodes(self, linkable):
        while linkable:
            print("Trying to connect waiting nodes to new nodes: ", linkable)
            if not self.waiting_nodes:
                return True
            nodes = []
            wei = []
            tw = 0
            for n in self.waiting_nodes:
                w = n.get_weight()
                if w > 0:
                    nodes.append(n)
                    wei.append(w)
                    tw += w
            sorted_wait_nodes = numpy.random.choice(nodes, len(nodes), False, [w / tw for w in wei])
            added_super = []
            for node in sorted_wait_nodes:
                visible_links = [link for link in self.check_connectivity(linkable, node.building) if link]
                if not visible_links:
                    continue
                is_leaf = node.cost_choice is CostChoice.LEAF_NODE
                res = self.add_links_leaf(node, visible_links) if is_leaf else self.add_links_super(node, visible_links)
                if res:
                    print("Successfully connected waiting node: ", node.building)
                    if self.stop_condition_maxnodes():
                        return True
                    self.waiting_nodes.remove(node)
                    if not is_leaf:
                        linkable.append(node.building)
                        added_super.append(node.building)
            linkable = added_super

    def load_nodes_from_buildings(self, buildings):
        for building in buildings:
            if building.gid == self.gw_node.gid:
                continue
            if building.gid in self.db_nodes:
                continue
            try:
                node = CostNode(self.args.max_dev, self.CI.get_cached(building))
            except CostError as e:
                print("Ignoring building %r cause error: %s" % (building, e))
                continue
            self.db_nodes[building.gid] = node
            self.susceptible.add(node)
