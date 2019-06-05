import unittest

import math
import numpy
from libterrain.building import Building
from shapely.geometry import Polygon

import network
import ubiquiti as ubnt
from cn_generator import CN_Generator
from cost_interface import PolyCostParam, AppliedCostModel, AppliedVolumeModel, IstatCostInterface
from cost_model import IstatCostModel, CostError, MetricType
from misc import BuildingUtils
from node import AntennasExahustion, ChannelExahustion, LinkUnfeasibilty, CostChoice, Node, CostNode
import numpy as np
import math as m

from strategies.cost_strategy import CostStrategy


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


class NetworkTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(NetworkTests, self).__init__(*args, **kwargs)
        self.n = network.Network()
        ubnt.load_devices()
        self.n.set_maxdev(4)

    def _calc_angles(self, src, trg):
        rel_pos = np.subtract(trg, src)
        yaw = m.atan2(rel_pos[1], rel_pos[0])
        pitch = m.atan2(rel_pos[2], m.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2))
        # yaw and pitch are in the range -pi - pi
        # lets add 180° (to avoid pi approx) to the degree to have them in the space
        # 0-360°
        return (m.degrees(yaw) + 180, m.degrees(pitch) + 180)

    def gen_link(self, src, dst):
        link = {}
        link['src'] = src
        link['dst'] = dst
        link['loss'] = 50  # Fixed
        link['src_orient'] = self._calc_angles(src.xyz(), dst.xyz())
        link['dst_orient'] = self._calc_angles(dst.xyz(), src.xyz())
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
        self.n.add_link_generic(self.gen_link(b3, b4))
        link = self.gen_link(b3, b1)
        # We must reverse the link because the antenna must be added on the dst, not the src
        self.n.add_link_generic(link, reverse=True)
        assert (len(self.n.graph.nodes[b3.gid]['node']) == 2)
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
        super().__init__()
        self.override_metrics = override_metrics
        self.override_std = override_std
        self.buildings = []

    def get_metrics(self):
        return super().get_metrics() if self.override_metrics is None else self.override_metrics

    def get_std_dev(self):
        return super().get_std_dev() if self.override_std is None else self.override_std

    def get_weight(self):
        return 1


def build_square_shape(center, area):
    edge = math.sqrt(area)
    x = center[0]
    y = center[1]
    return Polygon([(x - edge / 2, y - edge / 2), (x - edge / 2, y + edge / 2), (x + edge / 2, y + edge / 2),
                    (x + edge / 2, y - edge / 2)])


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

    def test_empty_istat_area(self):
        """Istat areas should always have at least one building"""
        model = IstatCostModel()
        with self.assertRaises(CostError):
            model.get_buildings_volume()
        model.buildings = []
        with self.assertRaises(CostError):
            model.get_buildings_number()

    def test_area_std_dev(self):
        model = IstatCostModel()
        with self.assertRaises(CostError):
            model.get_std_dev()
        model.buildings = build_building_array([10, 10, 10, 10, 10])
        self.assertAlmostEqual(model.get_std_dev(), 0)
        model.buildings = build_building_array([110, 10])
        model.buildings_std = None
        self.assertAlmostEqual(model.get_std_dev(), numpy.std([110, 10]) / numpy.average([110, 10]), 5)

    def test_buildings_metrics(self):
        model = IstatCostModel()
        model.buildings = build_building_array([10, 10, 10, 10, 10])
        self.assertEqual(model.get_buildings_number(), 5)
        self.assertAlmostEqual(model.get_buildings_volume() / 50, 1, 5)
        model.buildings = build_building_array([100, 1, 3000])
        model.buildings_volume = None
        self.assertEqual(model.get_buildings_number(), 3)
        self.assertAlmostEqual(model.get_buildings_volume() / 3101, 1, 5)


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
    def test_get_metrics(self):
        building = FakeVolumeBuilding(0, (0, 0), 100, 1)
        met = {"test_param": (100, MetricType.UNIQUE), "test_param_2": (250, MetricType.UNIQUE)}
        model = FakeIstatModel(None, met)
        applied_met = AppliedCostModel(building, model, {}).applied_metrics()
        for key, value in met.items():
            self.assertEqual(value[0], applied_met[key])
        with self.assertRaises(CostError):
            applied = AppliedVolumeModel(building, model, {})
        model.buildings = build_building_array([100, 200, 600, 100])
        model.buildings_volume = None
        applied = AppliedVolumeModel(building, model, {})
        applied_met = applied.applied_metrics()
        for key, value in met.items():
            self.assertAlmostEqual((value[0] * (100 / 1000)) / applied_met[key], 1, 5)
        building = FakeVolumeBuilding(0, (0, 0), 600, 1)
        applied_met = AppliedVolumeModel(building, model, {}).applied_metrics()
        for key, value in met.items():
            self.assertAlmostEqual((value[0] * (600 / 1000)) / applied_met[key], 1, 5)

    def test_std_mean(self):
        building = FakeVolumeBuilding(0, (0, 0), 100, 1)
        for mval in [(100, 250, 60), (500, 1000, 8), (6, 0, -7), (501, -9999, 55)]:
            a = mval[0]
            b = mval[1]
            c = mval[2]
            met = {"test_param": (a, MetricType.UNIQUE), "test_param_2": (b, MetricType.UNIQUE),
                   "test_param_3": (c, MetricType.UNIQUE)}
            for points in [(0, 10, 0, 20, 4, -1), (100, 20, 20, 0, 54, 45), (1000, 10, -9999, 88, 9, 12),
                           (123, -23, 0, 0, 0, 0)]:
                x = points[0]
                y = points[1]
                x2 = points[2]
                y2 = points[3]
                x3 = points[4]
                y3 = points[5]
                model = FakeIstatModel(0, met)
                avg, std = AppliedCostModel(building, model,
                                            {"test_param": PolyCostParam([(x, y)])}).get_mean_std_percentage()
                self.assertAlmostEqual(avg, y)
                self.assertAlmostEqual(std, 0)
                avg, std = AppliedCostModel(building, model,
                                            {"test_param": PolyCostParam([(x, y), (x + 10, y + 10)]),
                                             "test_param_2": PolyCostParam([(x2, y2)])}).get_mean_std_percentage()
                aavg, astd = AppliedCostModel._weighted_avg_std([y - x + a, y2], [1, 1])
                self.assertAlmostEqual(avg, aavg)
                self.assertAlmostEqual(std, astd)
                avg, std = AppliedCostModel(building, model,
                                            {"test_param": PolyCostParam([(x, y)], weight=8),
                                             "test_param_2": PolyCostParam([(x2, y2)],
                                                                           weight=2)}).get_mean_std_percentage()
                self.assertGreaterEqual(avg, min(y, y2))
                self.assertLessEqual(avg, max(y, y2))
                avg, std = AppliedCostModel(building, model,
                                            {"test_param": PolyCostParam([(x, y), (1, 9)], weight=0),
                                             "test_param_2": PolyCostParam([(x2, y2)],
                                                                           weight=2)}).get_mean_std_percentage()
                self.assertAlmostEqual(avg, y2)
                self.assertAlmostEqual(std, 0)
                with self.assertRaises(ValueError):
                    avg, std = AppliedCostModel(building, model,
                                                {"test_param": PolyCostParam([(x, y)], weight=-3),
                                                 "test_param_2": PolyCostParam([(x2, y2)],
                                                                               weight=2)}).get_mean_std_percentage()
                avg, std = AppliedCostModel(building, model,
                                            {"test_param": PolyCostParam([(x, y)], weight=0),
                                             "test_param_2": PolyCostParam([(x2, y2)], weight=2),
                                             "test_param_3": PolyCostParam(
                                                 [(x3, y3), (x3 + 1, y3 + 1)])}).get_mean_std_percentage()
                self.assertGreaterEqual(avg, min(y, y2, y3 - x3 + c))
                self.assertLessEqual(avg, max(y, y2, y3 - x3 + c))


class FakeIstatInterface(IstatCostInterface):
    def __init__(self, model, params):
        self.test_model = model
        self.cache = {}
        self.use_cache = True
        self.apply_model = AppliedVolumeModel
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


class TestIstatInterface(unittest.TestCase):
    # model
    huge_model = FakeIstatModel(None, {"test_param": (3000, MetricType.UNIQUE), "test_param2": (40, MetricType.UNIQUE)})
    tiny_model = FakeIstatModel(None,
                                {"test_param": (-3000, MetricType.UNIQUE), "test_param2": (40, MetricType.UNIQUE)})
    interface = FakeIstatInterface(huge_model, {"test_param": PolyCostParam([(0, 0), (1, 1)]),
                                                "test_param2": PolyCostParam([(0, 0)])})
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


if __name__ == '__main__':
    unittest.main()
