import numpy

from cost_model import CostError
from node import CostChoice
from strategies.cost_strategy import CostStrategy


class CostStrategyAggressive(CostStrategy):



    def __init__(self, args, unk_args={}):
        super().__init__(args, unk_args)
        sort = int(getattr(args, 'sort_unattached', 3)) % 4
        self.sort_alg = sort % 2
        self.sort_type = 0 if sort < 2 else 1

    def add_links(self, new_node):
        res = super().add_links(new_node)
        if res and new_node.cost_choice is CostChoice.SUPER_NODE:
            self.add_linkable_nodes([new_node])
        return res

    def sort_nodes_to_attach(self, nodes):
        if self.sort_alg == 1:
            sortable = []
            wei = []
            tw = 0
            for n in nodes:
                w = n.get_weight()
                if w > 0:
                    sortable.append(n)
                    wei.append(w)
                    tw += w
            sorted_nodes = numpy.random.choice(sortable, len(sortable), False, [w / tw for w in wei])
        else:
            sorted_nodes = [n for n in nodes]
        if self.sort_type == 1:
            res = [n for n in sorted_nodes if n.cost_choice is CostChoice.SUPER_NODE] + [n for n in sorted_nodes if n.cost_choice is not CostChoice.SUPER_NODE]
        else:
            res = sorted_nodes
        return res

    def add_linkable_nodes(self, linkable):
        while linkable:
            if not self.waiting_nodes:
                return True
            print("Trying to connect waiting nodes to new nodes: ", linkable)
            sorted_wait_nodes = self.sort_nodes_to_attach(self.waiting_nodes)
            added_super = []
            for node in sorted_wait_nodes:
                visible_links = [link for link in self.check_connectivity(linkable, node) if link]
                if not visible_links:
                    continue
                is_leaf = node.cost_choice is CostChoice.LEAF_NODE
                res = self.add_links_leaf(node, visible_links) if is_leaf else self.add_links_super(node, visible_links)
                if res:
                    print("Successfully connected waiting node: ", node)
                    self.waiting_nodes.remove(node)
                    self.restructure()
                    if self.stop_condition():
                        return True
                    if not is_leaf:
                        added_super.append(node)
                        linkable.append(node)
                else:
                    self.add_node(node, False)
            linkable = added_super

    def load_nodes_from_buildings(self, buildings):
        susceptibles = set(buildings)-self.ignored
        for building in susceptibles:
            if building.gid == self.gw_node.building.gid:
                continue
            if building.gid in self.db_nodes:
                continue
            nodes = self.gen_nodes_from_building(building)
            if nodes is None:
                continue
            self.db_nodes[building.gid] = nodes
            self.susceptible.update(nodes)
