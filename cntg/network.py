import networkx as nx
from antenna import Antenna
import ubiquiti as ubnt
import numpy

node_fixed_cost = 200


def compute_link_quality(left, right, attrs, min_rate=6):
    """ We want to express metrics as a cost, so high=bad, 1/bandwidth may
    introduce non linearities with non additive metrics, thus we rescale it for
    a basic minimum rate  """
    return min_rate * attrs['link_per_antenna'] / attrs['rate']


class Network():
    def __init__(self):
        self.graph = nx.DiGraph()
        self.cost = 0

    def set_maxdev(self, max_dev):
        self.max_dev = max_dev

    def size(self):
        return len(self.graph)

    def main_sg(self):
        # Return the connected subgraph belonging to the gateway
        # Always return something, otherwise something very bad is happening
        sgs = nx.connected_component_subgraphs(self.graph.to_undirected())
        for sg in sgs:
            if self.gateway in sg:
                return sg

    def add_gateway(self, building, attrs={}):
        self.add_node(building, attrs)
        self.gateway = building.gid

    def add_node(self, building, attrs={}):
        self.graph.add_node(building.gid, pos=building.xy(), antennas=list(),
                            cost=node_fixed_cost, **attrs)
        self.cost += node_fixed_cost

    def del_node(self, building):
        self.graph.remove_node(building.gid)
        self.cost -= node_fixed_cost

    def add_link(self, link, existing_antenna=None, attrs={}):
        ant1 = None
        device0 = None
        link_per_antenna = 2
        # loop trough the antennas of the node
        for antenna in self.graph.nodes[link['dst'].gid]['antennas']:
            ant1 = antenna
            if existing_antenna and existing_antenna.channel != ant1.channel:
                continue
            if antenna.check_node_vis(link_angles=link['dst_orient']):
                rate0, device0 = ubnt.get_fastest_link_hardware(
                                    link['loss'],
                                    target=antenna.ubnt_device[0])
                if not rate0:
                    # no radio available for this (link loss, target antenna)
                    # try with a better antenna
                    # TODO: We can have multiple antenna in the same viewshed
                    # -> must doublecheck it
                    # Now we select the first one we find, but we should search
                    # for the optimal one (max rate)
                    ant1 = None
                    continue
                rate0, rate1 = ubnt.get_maximum_rate(link['loss'], device0[0],
                                                     antenna.ubnt_device[0])
                # TODO: check any link connected with ant1, find the sharing
                # factor, add one to its sharing factor, store the sharing
                # factor in the new link
                for l in self.graph.out_edges(link['dst'].gid, data=True):
                    if l[2]['src_ant'] == antenna:
                        link_per_antenna = l[2]['link_per_antenna'] + 2
                        l[2]['link_per_antenna'] += 2
                for l in self.graph.in_edges(link['dst'].gid, data=True):
                    if l[2]['dst_ant'] == antenna:
                        link_per_antenna = l[2]['link_per_antenna'] + 2
                        l[2]['link_per_antenna'] += 2
                break
        if existing_antenna:
            # if i want to add an existing antenna on src only to existing
            # antennas on dst 
            if ant1:
                # if i found it
                self.graph.add_edge(link['src'].gid, link['dst'].gid, loss=link['loss'],
                                    src_ant=ant0, dst_ant=ant1, rate=rate0,
                                    link_per_antenna=link_per_antenna, **attrs)
                self.graph.add_edge(link['dst'].gid, link['src'].gid, loss=link['loss'],
                                    src_ant=ant1, dst_ant=ant0, rate=rate1,
                                    link_per_antenna=link_per_antenna, **attrs)
                return True
            else:
                # no found antenna no connection
                return None
        if not ant1:
            # Antenna not found so we add it
            n_ant = len(self.graph.nodes[link['dst'].gid]['antennas'])
            if(n_ant >= self.max_dev):
                # Antenna can't be added because we reached the saturtion, no link.
                return False
            rate1, device1 = ubnt.get_fastest_link_hardware(link['loss'])
            if not rate1:
                # TODO: What should we do if the link is unfeasible?
                return False
            rate0, rate1 = ubnt.get_maximum_rate(link['loss'], device1[0],
                                                 device1[0])
            device0 = device1
            ant1 = self.add_antenna(link['dst'].gid, device1, link['dst_orient'])

        # find proper device add antenna to local node
        ant0 = self.add_antenna(link['src'].gid, device0, link['src_orient'], ant1.channel)
        self.graph.add_edge(link['src'].gid, link['dst'].gid, loss=link['loss'],
                            src_ant=ant0, dst_ant=ant1, rate=rate0,
                            link_per_antenna=link_per_antenna, **attrs)
        self.graph.add_edge(link['dst'].gid, link['src'].gid, loss=link['loss'],
                            src_ant=ant1, dst_ant=ant0, rate=rate1,
                            link_per_antenna=link_per_antenna, **attrs)
        return ant0

    def add_antenna(self, node, device, orientation, channel=0):
        if not channel:
            free_channels = wifi.channels[:]
            for ant in self.graph.nodes[node]['antennas']:
                a.remove(ant.channel)
            try:
                channel = random.sample(free_channel, 1)
            except ValueError:
                # more antennas than channels, we have to re-use
                # this should not happen, or else we have to divide
                # capacity for all links in the same channel
                print("wARN: you have more antennas than available channels")
                channel = random.sample(wifi.channels, 1)

        ant = Antenna(device, orientation, channel)
        self.graph.nodes[node]['antennas'].append(ant)
        self.graph.nodes[node]['cost'] += ant.device['average_price']
        self.cost += ant.device['average_price']
        return ant

    def save_graph(self, filename):
        self.compute_minimum_bandwidth()
        for node in self.graph:
            self.graph.node[node]['x'] = self.graph.node[node]['pos'][0]
            self.graph.node[node]['y'] = self.graph.node[node]['pos'][1]
            del self.graph.node[node]['pos']
            # remove antenna to allow graph_ml exportation
            del self.graph.node[node]['antennas']
        for edge in self.graph.edges():
            del self.graph.edges[edge]['src_ant']
            del self.graph.edges[edge]['dst_ant']
        nx.write_graphml(self.graph, filename)

    def compute_minimum_bandwidth(self):
        min_bandwidth = {}
        for d in self.graph.nodes():
            if d == self.gateway:
                continue
            try:
                rev_path = nx.dijkstra_path(self.graph, d,
                                            self.gateway,
                                            weight=compute_link_quality)
            except nx.exception.NetworkXNoPath:
                continue
            min_b = float('inf')
            for i in range(len(rev_path) - 1):
                attrs = self.graph.get_edge_data(rev_path[i], rev_path[i + 1])
                b = attrs['rate'] / attrs['link_per_antenna']
                if b < min_b:
                    min_b = b
            self.graph.node[d]['min_bw'] = min_b
            min_bandwidth[d] = min_b
        return min_bandwidth

    def compute_equivalent_connectivity(self):
        """ if C_0 is the component including the gateway, then if graph
        connectivity > 1, return 1/connectivity, else return the number of
        cut-points """
        if len(self.graph) < 2:
            return 1
        main_comp = None
        for c in nx.connected_component_subgraphs(self.graph.to_undirected()):
            if self.gateway in c.nodes():
                main_comp = c

        connectivity = nx.node_connectivity(main_comp)
        if connectivity >= 1:
            return 1/connectivity
        else:
            return len(list(nx.articulation_points(
                               main_comp.to_undirected())))

    def compute_metrics(self):
        """ returns a dictionary with a set of metrics that evaluate the
        network graph under several points of view """

        metrics = {}
        min_bandwidth = self.compute_minimum_bandwidth()
        disconnected_nodes = 0
        percentiles = [10, 50, 90]
        for perc in percentiles:
            metrics["perc_"+str(perc)] = ""

        counter = 1
        per_i = 0
        for d, b in sorted(min_bandwidth.items(), key=lambda x: x[1]):
            if not b:
                disconnected_nodes += 1
            else:
                perc = int(100*counter/len(min_bandwidth))
                counter += 1
                if per_i < len(percentiles) and perc > percentiles[per_i]:
                    metrics["perc_"+str(percentiles[per_i])] = b
                    per_i += 1

        metrics["connected_nodes"] = 1 + len(min_bandwidth) -\
                                       disconnected_nodes
        metrics["unconnected_ratio"] = disconnected_nodes / \
                                       (1 + len(min_bandwidth))
        metrics["price_per_user"] = self.cost/metrics['connected_nodes']
        metrics["price_per_mbyte"] = 8*metrics['price_per_user'] * \
                                     sum([1/x for x in min_bandwidth.values()
                                         if x])/metrics['connected_nodes']
        metrics["cut_points"] = 1/self.compute_equivalent_connectivity()
        # more useful metrics
        metrics["avg_link_per_antenna"] = numpy.mean([d['link_per_antenna']
                                                     for _, _, d in
                                                     self.graph.edges(
                                                     data=True)])
        return metrics
