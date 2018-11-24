from libterrain.libterrain import terrain
from geoalchemy2.shape import to_shape
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from multiprocessing import Pool
from misc import NoGWError
import shapely
import random
import time
import networkx as nx
import argparse
import network
import folium
from folium import plugins
import ubiquiti as ubnt
from edgeffect import EdgeEffect
import multiprocessing as mp
import os
import psutil
import datetime

def poor_mans_color_gamma(bitrate):
    blue_to_red = {200: '#03f', 150: '#6600cc', 100: '#660099',
                   50: '#660066', 30: '#660000'}
    for b in sorted(blue_to_red):
        if bitrate < b:
            return blue_to_red[b]
    return blue_to_red[200]


class CN_Generator():

    DSN = "postgresql://dbreader@192.168.160.11/terrain_ans"

    def __init__(self, dataset, DSN=None, args={}, unk_args={}):
        self.round = 0
        self.infected = {}
        self.susceptible = set()
        self.pool = None
        self.net = network.Network()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-D", help="debug: print metrics at each iteration"
                                 " and save metrics in the given debug file",
                                 nargs='?', default="", const="")
        self.parser.add_argument("-P", help="number of parallel processes",
                                 default=1, type=int)
        self.parser.add_argument("-p", help="plot the graph using the browser",
                                 dest='plot', action='store_true')
        self.parser.add_argument('-b',
                                 help="gateway latlong (lat.dd,long.dd)",
                                 type=str, required=True)
        self.parser.add_argument('-n', help="number of nodes", type=int)
        self.parser.add_argument('-e', help="expansion range (in meters),"
                                 " defaults to buildings at 30km", type=float,
                                 default=30000)
        self.parser.add_argument('-r', help="random seed,", type=int,
                                 default=1)
        self.parser.add_argument('-B', help="Accepts two arguments: bw frac."
                "Stop when a fraction of frac nodes has less than bw bandwidth"
                "(in Mbps). Ex: '1 0' will stop when any node has less than 1Mbps",
                type=float, default=[1, 0], nargs=2)
        self.parser.add_argument('-R', help="restructure with edgeffect every r"
                " rounds, adding l links. Accepts two arguments: r l",)
        self.parser.add_argument('-V', help="Add at most v links extra link if"
                "these are in the viewshed of the current one.", type=int, default=0)
        self.args = self.parser.parse_args(unk_args)
        self.n = self.args.n
        self.e = self.args.e
        self.b = self.args.b
        self.P = self.args.P
        self.B = self.args.B
        self.R = self.args.R
        self.V = self.args.V
        self.random_seed = self.args.r
        self.debug_file = None
        random.seed(self.random_seed)
        self.net.set_maxdev(args.max_dev)
        self.parser.set_defaults(plot=False)
        if not DSN:
            self.t = terrain(self.DSN, dataset, ple=2.4, processes=self.P)
        else:
            self.t = terrain(DSN, dataset, ple=2.4, processes=self.P)
        self.event_counter = 0
        ubnt.load_devices()

    def _post_init(self):
        latlong = self.b.split(",")
        self.gw_pos = Point(float(latlong[1]), float(latlong[0]))
        gateway = self.get_gateway()
        self.infected[gateway.gid] = gateway
        self.net.add_gateway(gateway, attrs={'event': 0})
        self.event_counter += 1
        self.get_susceptibles()
        print("The gateway is " + repr(gateway))

    def get_gateway(self):
        buildings = self.t.get_buildings(shape=self.gw_pos)
        if len(buildings) < 1:
            raise NoGWError
        return buildings[0]

    def get_random_node(self):
        #must cast into list and order because sample on set is unpredictable
        susceptible_tmp = sorted(list(self.susceptible), key=lambda x:x.gid)
        new_node = random.sample(susceptible_tmp, 1)[0]
        self.susceptible.remove(new_node)
        return new_node

    def get_susceptibles(self):
        geoms = [g.shape() for g in self.infected.values()]
        self.sb.set_shape(geoms)
        db_buildings = self.t.get_buildings(self.sb.get_buffer(self.e))
        self.susceptible = set(db_buildings) - set(self.infected.values())

    def get_newnode(self):
        raise NotImplementedError

    def stop_condition(self):
        raise NotImplementedError

    def stop_condition_maxnodes(self):
        return len(self.infected) > self.n

    def stop_condition_minbw(self, rounds=1):
        #in case you don't want to test the stop condition every round
        if len(self.infected) % rounds != 0:
            return False

        # recompute minimum bw at each node
        bw = self.B[0]
        self.net.compute_minimum_bandwidth()
        # if the minimum bw of a node is less than the treshold stop
        self.below_bw_nodes = 0
        for n in self.infected:
            if n == self.net.gateway:
                continue
            try:
                if self.net.graph.node[n]['min_bw'] < bw:
                    self.below_bw_nodes += 1
                    if self.below_bw_nodes/len(self.infected) > self.B[1]:
                        return True
            except KeyError:
                #if the nod has no 'min_bw' means that it is not connected
                pass
        return False

    def add_links(self, new_node):
        raise NotImplementedError

    def check_connectivity(self, nodes, new_node):
        links = self.t.get_link_parallel(source_b=new_node, 
                                         dst_b_list=nodes)
        return links

    def restructure(self):
        raise NotImplementedError

    def main(self):
        try:
            while not self.stop_condition():
                self.round += 1
                # pick random node
                new_node = self.get_newnode()
                # connect it to the network
                if(self.add_links(new_node)):
                    # update area of susceptible nodes
                    self.get_susceptibles()
                    self.restructure()
                    print("Number of nodes:%d, infected:%d, susceptible:%d, "
                          "Nodes below bw:%d"
                          % (self.net.size(), len(self.infected),
                             len(self.susceptible), self.below_bw_nodes))
                    if self.args.D and len(self.net.graph) > 2:
                        self.print_metrics()
                        self.plot_map()
                    #input("stop me")
        except KeyboardInterrupt:
            pid = os.getpid()
            killtree(pid)
            pass
        # save result
        min_b = self.net.compute_minimum_bandwidth()
        for k, v in self.net.compute_metrics().items():
            print(k, v)
        if self.args.plot:
            self.save_evolution()
            self.plot_map()
            print("A browsable map was saved in " + self.map_file)
            print("A browsable animated map was saved in " +
                  self.animation_file)
        if self.debug_file:
            self.debug_file.close()#close(self.f)

    def restructure_edgeeffect_mt(self, num_links=1):
        # run only every self.args.R[0] nodes added
        if not self.args.R or self.net.size() % self.args.R[0] != 0:
            return

        num_links = self.R[1]
        max_links = num_links
        ee = EdgeEffect(self.net.graph, self.net.main_sg())
        if not self.pool:
            self.pool = Pool(self.P)
        effect_edges = self.pool.map(ee.restructure_edgeeffect, self.feasible_links)
        effect_edges.sort(key=lambda x: x['effect'])
        # Try to connect the best link (try again till it gets connected)
        while(effect_edges):
            selected_edge = effect_edges.pop()
            link = [link for link in self.feasible_links
                    if link['src'].gid == selected_edge[0] and
                    link['dst'].gid == selected_edge[1]
                    ]
            try:
                self.add_link(link[0], existing=True)
            except (LinkUnfeasibilty, AntennasExahustion, ChannelExahustion):
                pass
            else:
                max_links -= 1
                if max_links <= 0:
                    print("Restructured {} links".format(num_links))
                    return

    def add_node(self, node):
        self.event_counter += 1
        return self.net.add_node(node, attrs={'event': self.event_counter})

    def add_link(self, link, existing=False, reverse=False):
        self.event_counter += 1
        return self.net.add_link_generic(link=link,
                                         attrs={'event': self.event_counter},
                                         existing=existing,
                                         reverse=reverse)

    def save_graph(self):
        self.net.save_graph(self.filename)

    def graph_to_animation(self):
        quasi_centroid = self.t.polygon_area.representative_point()
        self.animation = folium.Map(location=(quasi_centroid.y,
                                    quasi_centroid.x),
                                    zoom_start=14, tiles='OpenStreetMap')
        p = shapely.ops.cascaded_union([pl for pl in self.t.polygon_area])
        point_list = list(zip(*p.exterior.coords.xy))
        folium.PolyLine(locations=[(y, x) for (x, y) in point_list],
                        fill_color="green", weight=1,
                        color='green').add_to(self.animation)
        edges_s = sorted(self.net.graph.edges(data=True),
                         key=lambda x: x[2]['event'])
        nodes_s = sorted(self.net.graph.nodes(data=True),
                         key=lambda x: x[1]['event'])
        last_event = max(edges_s[-1][2]['event'], nodes_s[-1][1]['event'])
        e_coords = []
        e_times = []
        for e in edges_s:
            e_coords.append([list(self.net.graph.nodes()[e[0]]['pos']),
                            list(self.net.graph.nodes()[e[1]]['pos'])])
            e_times.append(1530744263666 + e[2]['event'] * 36000000)
            # FIXME starting time is just a random moment
            features_edges = {
                'type': 'Feature',
                'geometry': {
                    'type': 'MultiLineString',
                    'coordinates': e_coords,
                },
                'properties': {
                    'times': e_times,
                }
            }
        n_coords = []
        n_times = []

        for n in nodes_s:
            n_coords.append([n[1]['pos'], n[1]['pos']])
            n_times.append(1530744263666 + n[1]['event'] * 36000000)
        # the only way I found to plot the nodes is pretend they are
        # one-point lines
        features_nodes = {
            'type': 'Feature',
            'geometry': {
                'type': 'MultiLineString',
                'coordinates': n_coords,
            },
            'properties': {
                'times': n_times,
                'style': {
                    'color': 'red',
                    'width': 20,
                }
            }
        }

        plugins.TimestampedGeoJson({
            'type': 'FeatureCollection',
            'features': [features_edges, features_nodes]},
            transition_time=500, auto_play=False).add_to(self.animation)

    def graph_to_leaflet(self):
        quasi_centroid = self.t.polygon_area.representative_point()
        self.map = folium.Map(location=(quasi_centroid.y, quasi_centroid.x),
                              zoom_start=14, tiles='OpenStreetMap')
        p = shapely.ops.cascaded_union([pl for pl in self.t.polygon_area])
        point_list = list(zip(*p.exterior.coords.xy))
        folium.PolyLine(locations=[(y, x) for (x, y) in point_list],
                        fill_color="green", weight=1,
                        color='green').add_to(self.map)
        max_event = max(nx.get_node_attributes(self.net.graph, 'event').values())
        for node in self.net.graph.nodes(data=True):
            (lat, lon) = node[1]['pos']
            label="Node: %d<br>Antennas:<br> %s" % (node[0], node[1]['node'])
            opacity = node[1]['event']/max_event
            if node[0] == self.net.gateway:
                folium.Marker([lon, lat],
                              icon=folium.Icon(color='red'),
                              popup=label
                              ).add_to(self.map)
            else:
                folium.CircleMarker([lon, lat],
                                    fill=True,
                                    popup=label,
                                    fill_opacity=opacity).add_to(self.map)
        for frm, to, p in self.net.graph.edges(data=True):
            lat_f, lon_f = nx.get_node_attributes(self.net.graph, 'pos')[frm]
            lat_t, lon_t = nx.get_node_attributes(self.net.graph, 'pos')[to]
            label = "Loss: %d dB<br>Rate: %d mbps<br>link_per_antenna: %d<br> src_orient %f <br> dst_orient %f" % \
                    (p['loss'], p['rate'], p['link_per_antenna'], p['src_orient'][0], p['dst_orient'][0])
            weight = 1 + 8/p['link_per_antenna']  # reasonable defaults
            color = poor_mans_color_gamma(p['rate'])
            folium.PolyLine(locations=[[lon_f, lat_f], [lon_t, lat_t]],
                            weight=weight, popup=label,
                            color=color).add_to(self.map)

    def plot_map(self):
        self.graph_to_leaflet()
        self.map_file = '/tmp/index.html'
        self.map.save(self.map_file)

    def save_evolution(self):
        self.graph_to_animation()
        self.animation_file = '/tmp/index-animation.html'
        self.animation.save(self.animation_file)

    def print_metrics(self):
        m = self.net.compute_metrics()
        if not self.debug_file:
            folder = "./data/"
            os.makedirs(folder, exist_ok=True)
            filename = folder + self.args.D + \
                       datetime.datetime.now().strftime("_%h_%d_%H_%M") + ".log"
            self.debug_file = open(filename, "w+")
            header_line = "#" + str(self.args)
            print(header_line, file=self.debug_file)
            print("nodes,", ",".join(m.keys()), file=self.debug_file)
        print(len(self.net.graph), ",",  ",".join(map(str, m.values())), 
              file=self.debug_file)



def killtree(pid, including_parent=False):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    if including_parent:
        parent.kill()
