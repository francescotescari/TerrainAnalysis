import random
import unittest
from collections import defaultdict
from multiprocessing.pool import Pool

import math
import numpy
from libterrain import BuildingInterface
from libterrain.building import Building
from shapely.geometry import Polygon, Point

import network
import ubiquiti as ubnt
import wifi
from cn_generator import CN_Generator
from cost_interface import PolyCostParam, IstatCostInterface, ConstCostInterface, CostInterface
from cost_model import IstatCostModel, CostError, AppliedModel
from misc import BuildingUtils, Susceptible_Buffer
from node import AntennasExahustion, ChannelExahustion, LinkUnfeasibilty, CostChoice, Node, CostNode
import numpy as np
import math as m

from strategies.cost_strategy import CostStrategy
from strategies.pref_attachment import Pref_attachment


def build_square_shape(center, area):
    edge = math.sqrt(area)
    x = center[0]
    y = center[1]
    return Polygon([(x - edge / 2, y - edge / 2), (x - edge / 2, y + edge / 2), (x + edge / 2, y + edge / 2),
                    (x + edge / 2, y - edge / 2)])


class FakeNode(Node):

    def __init__(self, building):
        super().__init__(building, 4, CostChoice.SUPER_NODE)

    def xyz(self):
        return self.building.xyz()


class FakeBuilding():
    def __init__(self, gid, xy, z=2):
        self.gid = gid
        self.pos = xy
        self.z = z

    def xy(self):
        return self.pos

    def xyz(self):
        return (self.pos[0], self.pos[1], self.z)

    def __repr__(self):
        return str(self.gid)

    def __str__(self):
        return self.__repr__()


class FakeShapedBuilding(FakeBuilding):
    def __init__(self, gid, xy, z=2):
        super().__init__(gid, xy, z)
        self.fake_shape = build_square_shape(xy, 0.5)

    def shape(self):
        return self.fake_shape


def _calc_angles(src, trg):
    rel_pos = np.subtract(trg, src)
    yaw = m.atan2(rel_pos[1], rel_pos[0])
    pitch = m.atan2(rel_pos[2], m.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2))
    # yaw and pitch are in the range -pi - pi
    # lets add 180° (to avoid pi approx) to the degree to have them in the space
    # 0-360°
    return (m.degrees(yaw) + 180, m.degrees(pitch) + 180)


class NetworkTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(NetworkTests, self).__init__(*args, **kwargs)
        self.n = network.Network()
        ubnt.load_devices()
        self.n.set_maxdev(4)

    def gen_link(self, src, dst):
        link = {}
        link['src'] = src
        link['dst'] = dst
        link['loss'] = 50  # Fixed
        link['src_orient'] = _calc_angles(src.xyz(), dst.xyz())
        link['dst_orient'] = _calc_angles(dst.xyz(), src.xyz())
        return link

    def test_onelink(self):
        b1 = FakeNode(FakeBuilding(1, (0, 0)))
        self.n.add_gateway(b1)
        b2 = FakeNode(FakeBuilding(2, (5, 5)))
        self.n.add_node(b2)
        link = self.gen_link(b2, b1)
        self.n.add_link_generic(link)
        # Check that the two antennas are aligned
        self.assertEqual((self.n.graph.nodes[b1.gid]['node'].antennas[0].orientation[0] + 180) % 360,
                         self.n.graph.nodes[b2.gid]['node'].antennas[0].orientation[0])
        # Check that the channel is the same
        self.assertEqual(self.n.graph.nodes[b1.gid]['node'].antennas[0].channel,
                         self.n.graph.nodes[b2.gid]['node'].antennas[0].channel)

    def test_twolink_in_viewshed(self):
        b1 = FakeNode(FakeBuilding(1, (-0.1, -0.1)))
        self.n.add_gateway(b1)
        b2 = FakeNode(FakeBuilding(2, (10, 10)))
        self.n.add_node(b2)
        link = self.gen_link(b2, b1)
        self.n.add_link_generic(link)
        b3 = FakeNode(FakeBuilding(3, (5.1, 5.1)))
        self.n.add_node(b3)
        self.n.add_link_generic(self.gen_link(b3, b2))
        # verify that there is only 1 antenna per node
        for n in self.n.graph.nodes(data=True):
            self.assertEqual(len(n[1]['node']), 1)
        for e in self.n.graph.out_edges(b2.gid, data=True):
            self.assertEqual(e[2]['link_per_antenna'], 4)
            self.assertEqual(len(e[2]['interfering_links']), 4)

    def test_twolink_lind_viewshed(self):
        b1 = FakeNode(FakeBuilding(1, (0, 0)))
        self.n.add_gateway(b1)
        b2 = FakeNode(FakeBuilding(2, (5, 5)))
        self.n.add_node(b2)
        link = self.gen_link(b2, b1)
        self.n.add_link_generic(link, existing=True)
        b3 = FakeNode(FakeBuilding(3, (10, 10)))
        self.n.add_node(b3)
        self.n.add_link_generic(self.gen_link(b3, b2), existing=True)
        # verify that there is only 1 antenna per node (2 for the node in between)
        self.assertEqual(len(self.n.graph.nodes[b1.gid]['node']), 1)
        self.assertEqual(len(self.n.graph.nodes[b2.gid]['node']), 2)
        self.assertEqual(len(self.n.graph.nodes[b3.gid]['node']), 1)

    def test_twoisland_merge(self):
        b1 = FakeNode(FakeBuilding(1, (1, 1)))
        self.n.add_gateway(b1)
        b2 = FakeNode(FakeBuilding(2, (1, 2)))
        self.n.add_node(b2)
        link = self.gen_link(b2, b1)
        self.n.add_link_generic(link)
        b3 = FakeNode(FakeBuilding(3, (2, 1)))
        self.n.add_node(b3)
        self.n.add_link_generic(self.gen_link(b3, b1))
        b4 = FakeNode(FakeBuilding(4, (3, 1)))
        b5 = FakeNode(FakeBuilding(5, (3, 2)))
        self.n.add_node(b5)
        self.n.add_link_generic(self.gen_link(b5, b2))
        self.n.add_node(b4)
        self.n.add_link_generic(self.gen_link(b4, b5))

        self.n.add_link_generic(self.gen_link(b4, b1), existing=True)
        self.assertEqual(len(self.n.graph.nodes[b1.gid]['node']), 3)
        self.assertEqual(len(self.n.graph.nodes[b2.gid]['node']), 2)
        self.assertEqual(len(self.n.graph.nodes[b3.gid]['node']), 1)
        self.assertEqual(len(self.n.graph.nodes[b4.gid]['node']), 2)
        self.assertEqual(len(self.n.graph.nodes[b5.gid]['node']), 2)

    def test_ant_exaustion(self):
        b1 = FakeNode(FakeBuilding(1, (0, 0)))
        b2 = FakeNode(FakeBuilding(2, (1, 0)))
        b3 = FakeNode(FakeBuilding(3, (0, 1)))
        b4 = FakeNode(FakeBuilding(4, (-1, 0)))
        b5 = FakeNode(FakeBuilding(5, (0, -1)))
        b6 = FakeNode(FakeBuilding(6, (1, 1)))

        self.n.add_gateway(b1)
        self.n.add_node(b2)
        self.n.add_link_generic(self.gen_link(b2, b1))
        self.n.add_node(b3)
        self.n.add_link_generic(self.gen_link(b3, b1))
        self.n.add_node(b4)
        self.n.add_link_generic(self.gen_link(b4, b1))
        self.n.add_node(b5)
        self.n.add_link_generic(self.gen_link(b5, b1))
        self.n.add_node(b6)
        with self.assertRaises(AntennasExahustion):
            self.n.add_link_generic(self.gen_link(b6, b1))

    def test_extralink(self):
        b1 = FakeNode(FakeBuilding(1, (1, 1)))
        b2 = FakeNode(FakeBuilding(2, (2, 2)))
        b3 = FakeNode(FakeBuilding(3, (1, 3)))
        b4 = FakeNode(FakeBuilding(4, (1, 2)))
        self.n.add_gateway(b1)
        self.n.add_node(b2)
        self.n.add_link_generic(self.gen_link(b2, b1))
        self.n.add_node(b3)
        self.n.add_link_generic(self.gen_link(b3, b2))
        self.n.add_node(b4)
        ant = self.n.add_link_generic(self.gen_link(b3, b4))
        link = self.gen_link(b3, b1)
        # We must reverse the link because the antenna must be added on the dst, not the src
        new_ant_needed = ant.channel not in b1.free_channels
        self.n.add_link_generic(link, reverse=True)
        if new_ant_needed:
            self.assertEqual(len(self.n.graph.nodes[b3.gid]['node']), 3)
            return
        self.assertEqual(len(self.n.graph.nodes[b3.gid]['node']), 2)
        for e in self.n.graph.out_edges(b3.gid, data=True):
            if e[1] in [b1.gid, b4.gid]:
                self.assertEqual(e[2]['link_per_antenna'], 4)


class TestDataStr(unittest.TestCase):
    ubnt.load_devices()

    def test_fastest_link(self):
        for p in range(80, 160, 10):
            print(ubnt.get_fastest_link_hardware(p))

    def test_feasible_modulations(self):
        mod_list = ubnt.get_feasible_modulation_list("AM-LiteBeam5ACGEN2",
                                                     "AM-LiteBeam5ACGEN2",
                                                     100)


class FakeIstatModel(IstatCostModel):
    def __init__(self, override_std=None, override_metrics=None):
        super().__init__(None)
        self.override_metrics = override_metrics
        self.override_std = override_std
        self.buildings = []

    def get_metrics(self):
        return super().get_metrics() if self.override_metrics is None else self.override_metrics

    def get_std_dev(self):
        return super().get_std_dev() if self.override_std is None else self.override_std

    def get_weight(self):
        return 1

    def _init_mapped_entry(self, mapped_entry):
        pass


def build_building_array(areas, heights=None):
    bls = []
    lh = len(heights) if heights is not None else 0
    spacing = 0
    for i in range(len(areas)):
        bls.append(FakeVolumeBuilding(i, (spacing, 0), areas[i], heights[i] if i < lh else 1))
        spacing += 10 + math.sqrt(areas[i])
    return bls


class FakeVolumeBuilding(Building):
    __abstract__ = True
    ant_height = 4

    def __init__(self, gid, xy, area, height):
        self.gid = gid
        self.area = area
        self.height = height
        self.shp = build_square_shape(xy, area)
        self.shp = BuildingUtils.project_to_4326(self)
        self.xy = xy

    def shape(self):
        return self.shp

    def get_height(self):
        return self.height


class TestIstatModel(unittest.TestCase):

    def test_fake_volume_building(self):
        self.assertAlmostEqual(BuildingUtils.get_building_volume(FakeVolumeBuilding(0, (0, 0), 100, 1)) / 100, 1, 5)
        self.assertAlmostEqual(BuildingUtils.get_building_volume(FakeVolumeBuilding(0, (2, 6), 6543, 1)) / 6543, 1, 5)


class TestCostParam(unittest.TestCase):
    def test_poly_cost_param(self):
        # y = 3
        param = PolyCostParam([(0, 3)])
        self.assertEqual(param.normalize(30), 3)
        self.assertEqual(param.normalize(14), 3)
        # y = x-1
        param = PolyCostParam([(1, 0), (2, 1)])
        self.assertAlmostEqual(param.normalize(6), 5)
        self.assertAlmostEqual(param.normalize(0), -1)
        # y = x(x-1)(x+1)
        param = PolyCostParam([(1, 0), (-1, 0), (4, 60), (-2, -6)])
        self.assertAlmostEqual(param.normalize(8), 504)
        self.assertAlmostEqual(param.normalize(0), 0)
        # y = x^2 (circa)
        param = PolyCostParam([(0, 0), (1, 1), (2, 3.5), (-1, 0.9), (100, 9978)], 2)
        self.assertAlmostEqual(param.normalize(50) / 2500, 1, 1)
        self.assertAlmostEqual(param.normalize(7) / 49, 1, 1)


class TestAppliedModel(unittest.TestCase):

    def test_std_mean(self):
        building = FakeVolumeBuilding(0, (0, 0), 100, 1)
        for mval in [(100, 250, 60), (500, 1000, 8), (6, 0, -7), (501, -9999, 55)]:
            a = mval[0]
            b = mval[1]
            c = mval[2]
            met = {"test_param": a, "test_param_2": b, "test_param_3": c}
            for points in [(0, 10, 0, 20, 4, -1), (100, 20, 20, 0, 54, 45), (1000, 10, -9999, 88, 9, 12),
                           (123, -23, 0, 0, 0, 0)]:
                x = points[0]
                y = points[1]
                x2 = points[2]
                y2 = points[3]
                x3 = points[4]
                y3 = points[5]
                model = FakeIstatModel(0, met)
                avg, std = AppliedModel(building, {"test_param": PolyCostParam([(x, y)])}, met,
                                        model_std=model.get_std_dev())._get_mean_std_percentage()
                self.assertAlmostEqual(avg, y)
                self.assertAlmostEqual(std, 0)
                avg, std = AppliedModel(building, {"test_param": PolyCostParam([(x, y), (x + 10, y + 10)]),
                                                   "test_param_2": PolyCostParam([(x2, y2)])}, met,
                                        model_std=model.get_std_dev())._get_mean_std_percentage()
                aavg, astd = AppliedModel._weighted_avg_std([y - x + a, y2], [1, 1])
                self.assertAlmostEqual(avg, aavg)
                self.assertAlmostEqual(std, astd)
                avg, std = AppliedModel(building, {"test_param": PolyCostParam([(x, y)], weight=8),
                                                   "test_param_2": PolyCostParam([(x2, y2)], weight=2)}, met,
                                        model_std=model.get_std_dev())._get_mean_std_percentage()
                self.assertGreaterEqual(avg, min(y, y2))
                self.assertLessEqual(avg, max(y, y2))
                avg, std = AppliedModel(building, {"test_param": PolyCostParam([(x, y), (1, 9)], weight=0),
                                                   "test_param_2": PolyCostParam([(x2, y2)], weight=2)}, met,
                                        model_std=model.get_std_dev())._get_mean_std_percentage()
                self.assertAlmostEqual(avg, y2)
                self.assertAlmostEqual(std, 0)
                with self.assertRaises(ValueError):
                    avg, std = AppliedModel(building, {"test_param": PolyCostParam([(x, y)], weight=-3),
                                                       "test_param_2": PolyCostParam([(x2, y2)], weight=2)}, met,
                                            model_std=model.get_std_dev())._get_mean_std_percentage()
                avg, std = AppliedModel(building, {"test_param": PolyCostParam([(x, y)], weight=0),
                                                   "test_param_2": PolyCostParam([(x2, y2)], weight=2),
                                                   "test_param_3": PolyCostParam([(x3, y3), (x3 + 1, y3 + 1)])}, met,
                                        model_std=model.get_std_dev())._get_mean_std_percentage()
                self.assertGreaterEqual(avg, min(y, y2, y3 - x3 + c))

                self.assertLessEqual(avg, max(y, y2, y3 - x3 + c))


class FakeIstatInterface(IstatCostInterface):
    def __init__(self, model, params):
        self.test_model = model
        self.cache = {}
        self.use_cache = True
        self.apply_model = IstatCostModel
        self.parameters = params

    def get_model(self, building):
        return self.test_model


class FakeCN_Generator(CN_Generator):
    def __init__(self, interface):
        self.CI = interface
        self.net = network.Network()
        self.net.gateway = 1000
        self.node_cache = {}
        self.args = lambda: None
        setattr(self.args, "max_dev", 5)


class FakeBuildingInterface:

    def __init__(self, buildings):
        self.fake_buildings = buildings

    def get_buildings(self, shape, area=None):
        return self.fake_buildings


def create_test_strategy(strategy, buildings=[], link_gen=None, gateway=None, cost_interface=None, max_dev=4,
                         base_folder="./out/", minbw=None, restructure=None, seed=1, channel_width=20, view_extra=0,
                         max_nodes=100000):
    class Dummy:
        def __init__(self):
            self.round = 0
            self.infected = {}
            self.susceptible = set()
            self.leaf_nodes = set()
            self.super_nodes = set()
            self.pool = Pool(1)
            self.net = network.Network()
            # self.args = args
            self.n = max_nodes
            self.e = 100
            self.args = lambda x: None
            self.args.restructure = None
            self.args.D = None
            self.args.plot = False
            self.below_bw_nodes = 0
            # self.b = self.args.gateway
            # self.P = self.args.processes
            self.feasible_links = []
            if minbw:
                self.B = tuple(map(float, minbw.split(' ')))
            if restructure:
                self.R = tuple(map(int, restructure.split(' ')))
            self.V = view_extra
            self.dataset = None
            wifi.default_channel_width = channel_width
            if not seed:
                self.random_seed = random.randint(1, 10000)
            else:
                self.random_seed = seed
            self.debug_file = None
            random.seed(self.random_seed)
            self.net.set_maxdev(max_dev)
            self.datafolder = base_folder + "data/"
            self.graphfolder = base_folder + "graph/"
            self.mapfolder = base_folder + "map/"
            self.BI = FakeBuildingInterface(buildings)
            # self.polygon_area = self.BI.get_province_area(self.dataset)
            self.event_counter = 0
            self.noloss_cache = defaultdict(set)
            ubnt.load_devices()
            self.show_level = 0
            self.db_nodes = {}
            self.waiting_nodes = set()
            self.gw = None
            self.ignored = set()
            self.sb = Susceptible_Buffer()
            self.polygon_area = Polygon([(20, 20), (20, -20), (-20, 20), (-20, -20)])
            self.max_dev = max_dev
            self.CI = cost_interface

    inst = Dummy()
    inst.__class__ = strategy

    def get_gateway(self):
        return gateway

    def check_connectivity(self, nodes, node):
        if link_gen is None:
            return []
        return [link_gen(node, n) for n in nodes]

    inst.check_connectivity = check_connectivity.__get__(inst)
    inst.get_gateway = get_gateway.__get__(inst)

    inst._post_init()
    return inst


class TestIstatInterface(unittest.TestCase):
    # model
    huge_model = FakeIstatModel(None, {"test_param": 3000, "test_param2": 40})
    tiny_model = FakeIstatModel(None,
                                {"test_param": -3000, "test_param2": 40})
    interface = FakeIstatInterface(huge_model, {"test_param": PolyCostParam([(0, 0), (1, 1)]),
                                                "test_param2": PolyCostParam([(0, 0)])})
    interface.use_cache = False
    cn_gen = FakeCN_Generator(interface)
    # set model buildings
    huge_model.buildings = build_building_array([10, 20, 30, 40, 50])
    tiny_model.buildings = build_building_array([10, 20, 30, 40, 50])

    def test_huge_thresholds(self):
        self.interface.test_model = self.huge_model
        # select building is first one
        for building in self.huge_model.buildings:
            # now test interface
            applied_model = self.interface.get(building)

            self.assertIsNotNone(applied_model)
            thresholds = applied_model.get_probabilities()
            # huge value param should set the threshold to join the network and become supernode at zero
            self.assertEqual(thresholds[0], 0)
            self.assertEqual(thresholds[1], 0)

    def test_tiny_thresholds(self):
        self.interface.test_model = self.tiny_model

        # select building is first one
        for building in self.tiny_model.buildings:
            # now test interface
            applied_model = self.interface.get(building)
            self.assertIsNotNone(applied_model)
            thresholds = applied_model.get_probabilities()
            # huge value param should set the threshold to join the network and become supernode at zero
            self.assertEqual(thresholds[0], 1)
            self.assertEqual(thresholds[1], 1)

    def test_cost_nodes(self):
        self.cn_gen.node_cache = {}
        self.interface.test_model = self.tiny_model
        for building in self.tiny_model.buildings:
            node = CostNode(4, self.interface.get(building))
            self.assertIsNotNone(node)
            self.assertIs(node.cost_choice, CostChoice.NOT_INTERESTED)

        self.interface.test_model = self.huge_model
        self.cn_gen.node_cache = {}
        for building in self.huge_model.buildings:
            node = CostNode(4, self.interface.get(building))
            self.assertIsNotNone(node)
            self.assertIs(node.cost_choice, CostChoice.SUPER_NODE)


class FakeChoiceBuilding(FakeShapedBuilding):
    def __init__(self, gid, xy, choice=None):
        super().__init__(gid, xy)
        self.choice = choice


class ChoiceAppliedModel(AppliedModel):
    def __init__(self, building, choice=None):
        super().__init__(building, {})
        self.choice = choice

    def get_probabilities(self):
        if self.choice is CostChoice.NOT_INTERESTED:
            return (1, 1)
        elif self.choice is CostChoice.LEAF_NODE:
            return (0, 1)
        else:
            return (0, 0)


class FakeCostNode(CostNode):
    def __init__(self, max_ant, building, choice=None):
        super().__init__(max_ant, ChoiceAppliedModel(building, choice))


class FakeChoiceInterface(CostInterface):

    def __init__(self):
        super().__init__(None)

    def get_model(self, building):
        pass

    def get(self, building):
        return ChoiceAppliedModel(building, building.choice)


class TestStrategy(unittest.TestCase):

    def buildings_square(self, n, choice=None):
        res = {}
        for i in range(0, n ** 2):
            x = i % n
            y = i - x
            res[i] = FakeChoiceBuilding(i, (x, y), choice)
        return res

    def get_links_gen(self, building_links=None):
        if building_links is None:
            return None
        elif building_links != 'all':
            def gen_link(src, dst):
                sgid = src.building.gid
                dgid = dst.building.gid
                if (sgid, dgid) in building_links:
                    loss = building_links[(sgid, dgid)]
                elif (dgid, sgid) in building_links:
                    loss = building_links[(dgid, sgid)]
                else:
                    return None
                link = {'src': src, 'dst': dst, 'loss': loss,
                        'src_orient': _calc_angles(src.building.xyz(), dst.building.xyz()),
                        'dst_orient': _calc_angles(dst.building.xyz(), src.building.xyz())}
                return link

            return gen_link
        else:
            def gen_link(src, dst):
                link = {'src': src, 'dst': dst, 'loss': 50,
                        'src_orient': _calc_angles(src.building.xyz(), dst.building.xyz()),
                        'dst_orient': _calc_angles(dst.building.xyz(), src.building.xyz())}
                return link

            return gen_link

    def test_strategy_node_connection(self):
        gateway = FakeNode(FakeShapedBuilding(1, (0, 0)))
        buildings = [FakeShapedBuilding(2, (1, 0))]
        cost_interface = ConstCostInterface(CostChoice.SUPER_NODE)
        links = {}
        links[(1, 2)] = 50
        strat = create_test_strategy(CostStrategy, gateway=gateway, buildings=buildings,
                                     cost_interface=cost_interface, link_gen=self.get_links_gen(links))
        strat.main()
        self.assertEqual(len(strat.net.graph.nodes), 2)
        buildings_map = self.buildings_square(5)
        gateway = FakeNode(buildings_map[13])
        buildings = [b for b in buildings_map.values()]
        links = 'all'
        strat = create_test_strategy(CostStrategy, gateway=gateway, buildings=buildings,
                                     cost_interface=cost_interface, link_gen=self.get_links_gen(links), max_dev=500)
        strat.main()
        self.assertEqual(len(strat.net.graph.nodes), 25)

    def test_strategy_node_type(self):
        buildings_map = self.buildings_square(3, CostChoice.LEAF_NODE)
        gateway = FakeCostNode(8, buildings_map[4], CostChoice.SUPER_NODE)
        buildings = [b for b in buildings_map.values()]
        links = self.get_links_gen('all')
        cost_interface = FakeChoiceInterface()
        cost_interface.use_cache = False
        strat = create_test_strategy(CostStrategy, gateway=gateway, buildings=buildings,
                                     cost_interface=cost_interface, link_gen=links)
        strat.main()
        self.assertEqual(len(strat.net.graph.edges), 0)
        buildings[2].choice = CostChoice.SUPER_NODE
        strat = create_test_strategy(CostStrategy, gateway=gateway, buildings=buildings,
                                     cost_interface=cost_interface, link_gen=links, max_dev=500)
        strat.main()
        self.assertEqual(len(strat.net.graph.nodes), 9)
        def get_link_sn_only(src, dst):
            return None if src.cost_choice is CostChoice.LEAF_NODE or dst.cost_choice is CostChoice.LEAF_NODE else links(src, dst)
        strat = create_test_strategy(CostStrategy, gateway=gateway, buildings=buildings,
                                     cost_interface=cost_interface, link_gen=get_link_sn_only, max_dev=500)
        strat.main()
        self.assertEqual(len(strat.net.graph.edges), 2)
        buildings[6].choice = CostChoice.SUPER_NODE
        strat = create_test_strategy(CostStrategy, gateway=gateway, buildings=buildings,
                                     cost_interface=cost_interface, link_gen=get_link_sn_only, max_dev=500)
        strat.main()
        self.assertEqual(len(strat.net.graph.edges), 4)

    def test_viewshed(self):
        buildings_map = self.buildings_square(3, CostChoice.LEAF_NODE)
        gateway = FakeCostNode(8, buildings_map[4], CostChoice.SUPER_NODE)
        buildings = [b for b in buildings_map.values()]
        links = self.get_links_gen('all')
        cost_interface = FakeChoiceInterface()
        cost_interface.use_cache = False
        buildings[0].choice = CostChoice.SUPER_NODE
        buildings[8].choice = CostChoice.SUPER_NODE
        def links_only_0_8(src, dst):
            if src.building is not buildings[0] and src.building is not buildings[8]:
                return None
            return links(src, dst)

        strat = create_test_strategy(CostStrategy, gateway=gateway, buildings=buildings,
                                     cost_interface=cost_interface, link_gen=links_only_0_8, max_dev=500)
        strat.main()
        self.assertEqual(len(strat.net.graph.edges), 4)
        strat = create_test_strategy(CostStrategy, gateway=gateway, buildings=buildings,
                                     cost_interface=cost_interface, link_gen=links_only_0_8, max_dev=500, view_extra=10)
        strat.main()
        self.assertEqual(len(strat.net.graph.edges), 6)

    def test_visible_links(self):
        buildings_map = self.buildings_square(3, CostChoice.SUPER_NODE)
        gateway = FakeCostNode(8, buildings_map[4], CostChoice.SUPER_NODE)
        buildings = [b for b in buildings_map.values()]
        links = {k: 50 for k in ((0,1), (1,2), (2,5), (5,8), (8,7), (7,6), (6,3), (3,4))}
        cost_interface = FakeChoiceInterface()
        cost_interface.use_cache = False
        strat = create_test_strategy(CostStrategy, gateway=gateway, buildings=buildings,
                                     cost_interface=cost_interface, link_gen=self.get_links_gen(links), max_dev=500)
        strat.main()
        self.assertEqual(len(strat.net.graph.edges), 16)
        buildings[7].choice = CostChoice.LEAF_NODE
        strat = create_test_strategy(CostStrategy, gateway=gateway, buildings=buildings,
                                     cost_interface=cost_interface, link_gen=self.get_links_gen(links), max_dev=500)
        strat.main()
        # print(strat.net.graph.edges)
        self.assertEqual(len(strat.net.graph.edges), 6)

    def test_leaf_connection(self):
        buildings_map = self.buildings_square(3, CostChoice.LEAF_NODE)
        gateway = FakeCostNode(8, buildings_map[4], CostChoice.SUPER_NODE)
        buildings = [b for b in buildings_map.values()]
        buildings[0].choice = CostChoice.SUPER_NODE
        buildings[2].choice = CostChoice.SUPER_NODE
        cost_interface = FakeChoiceInterface()
        cost_interface.use_cache = False
        links = 'all'
        strat = create_test_strategy(CostStrategy, gateway=gateway, buildings=buildings,
                                     cost_interface=cost_interface, link_gen=self.get_links_gen(links), max_dev=500)
        strat.main()
        self.assertEqual(len(strat.net.graph.edges), 16)
        links = {(a,b): (50 if a == 0 or b == 0 else 80) for a in range(9) for b in range(9)}
        strat = create_test_strategy(CostStrategy, gateway=gateway, buildings=buildings,
                                     cost_interface=cost_interface, link_gen=self.get_links_gen(links), max_dev=500)

        strat.get_susceptibles()
        n1 = [n for n in strat.susceptible if n.building.gid == 0][0]
        n2 = [n for n in strat.susceptible if n.building.gid == 2][0]
        strat.add_links(n1)
        strat.add_links(n2)
        strat.get_susceptibles()
        strat.main()
        l = len(strat.net.graph.edges)
        self.assertTrue(l == 16 or l == 18)
        self.assertEqual(len(strat.net.graph.out_edges(FakeCostNode(4, buildings[0]).gid)), l/2-1) # 7/8 links = link to each leaf node (6) + link to gw (+link to other sn)
        self.assertEqual(len(strat.net.graph.out_edges(FakeCostNode(4, buildings[2]).gid)), l/2-7) # 1/2 link = link to gw (+ link to other sn)



if __name__ == '__main__':
    unittest.main()
