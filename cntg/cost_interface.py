from sqlalchemy.orm import sessionmaker
import numpy
from cost_model import IstatCostModel, CostError, AppliedModel

from node import CostChoice


class CostInterfaceError(CostError):
    """Raised from cost-related function errors"""
    pass


class CostInterface:
    parameters = {}
    ap_model = AppliedModel

    def __init__(self, cost_model):
        self.cache = {}
        self.use_cache = True
        self.cost_model = cost_model
        self.mapper = None

    def get_model(self, building):
        raise NotImplementedError

    def get_cached(self, building):
        if not self.use_cache:
            return self.get(building)
        if building.gid not in self.cache.keys():
            self.cache[building.gid] = self.get(building)
        return self.cache[building.gid]

    def get(self, building):
        model = self.get_model(building)
        if model is None:
            raise CostInterfaceError("Failed to get cost model of building %r" % building)
        return self.ap_model(building, self.parameters, model.get_metrics_cached(), model.get_properties(),
                             model.get_std_dev(), model.get_weight())


class IstatDataMapper:
    pass


class IstatCostInterface(CostInterface):

    def __init__(self, bi, eco_table_name=None):
        super().__init__(IstatCostModel)
        if not hasattr(bi.building_class, 'istat_data'):
            if eco_table_name is None:
                eco_table_name = 'eco_data_' + bi.building_class.__tablename__
            bi.add_economics_data(eco_table_name, IstatDataMapper, 'istat_data')
        self.bi = bi
        self.parameters["people"] = PolyCostParam([(0.5, 0), (12, 1)])
        self.parameters["avg_age"] = SegmentCostParam(
            [(0, -1), (10, 0), (17, 0.6), (20, 0.7), (23, 1), (33, 1.5), (40, 1), (50, 0.9), (60, 0.7), (70, 0.5),
             (100, 0.2)], 0.5)

    def get_model(self, building):
        mapped_entry = building.istat_data
        if mapped_entry is None:
            raise CostInterfaceError("Missing istat data entry for building %r " % building)
        return self.cost_model(mapped_entry)


class ConstAppliedModel(AppliedModel):
    def __init__(self, building, choice):
        self.choice = choice
        super().__init__(building, {}, metrics={}, properties={'Choice': choice}, model_std=0, weight=1)

    def get_probabilities(self):
        if self.choice is CostChoice.NOT_INTERESTED:
            return 1, 1
        elif self.choice is CostChoice.LEAF_NODE:
            return 0, 1
        elif self.choice is CostChoice.SUPER_NODE:
            return 0, 0
        else:
            raise CostInterfaceError("Unknown choice: %s" % self.choice)


class ConstCostInterface(CostInterface):

    def get_model(self, building):
        return None

    def __init__(self, cost_choice):
        super().__init__(None)
        self.choice = cost_choice

    def get(self, building):
        return ConstAppliedModel(building, self.choice)


class CostParam:
    def __init__(self, weight=1.0):
        self.weight = weight

    def normalize(self, value):
        raise NotImplementedError


class PolyCostParam(CostParam):
    def normalize(self, value):
        return numpy.polyval(self.polynomial, value)

    def __init__(self, points, degree=-1, weight=1.0):
        super().__init__(weight)
        x, y = [], []
        for point in points:
            x.append(point[0])
            y.append(point[1])
        if degree < 0:
            degree = len(points) - 1
        self.polynomial = numpy.polyfit(x, y, degree)


class SegmentCostParam(CostParam):
    def normalize(self, value):
        f = None
        for key, val in self.funcs.items():
            f = val
            if value < key:
                break
        assert f is not None
        return f.normalize(value)

    def __init__(self, points, weight=1.0):
        super().__init__(weight)
        if len(points) < 2:
            raise CostInterfaceError("Minimum of two points are necessary in segment cost param")
        last_point = points[0]
        funcs = {}
        for i in range(1, len(points)):
            p = points[i]
            funcs[p[0]] = PolyCostParam([last_point, p])
            last_point = p
        self.funcs = funcs


class ExpCostParam(CostParam):

    def __init__(self, points, shift=-1, weight=1.0):
        super().__init__(weight=weight)
        x, y = [], []
        for point in points:
            a = point[1] - shift
            if a <= 0:
                raise CostInterfaceError("Invalid exp point: (%d, %d)" % (point[0], point[1]))
            x.append(point[0])
            y.append(a)
        r = numpy.polyfit(x, numpy.log(y), 1, w=numpy.sqrt(y))
        self.a = numpy.exp(r[1])
        self.b = r[0]
        self.shift = shift

    def normalize(self, value):
        return self.a * numpy.exp(self.b * value) + self.shift


class WinsorizeCostParam(CostParam):

    def __init__(self, param, min=None, max=None, weight=1.0):
        super().__init__(weight=weight)
        self.param = param
        self.min = min
        self.max = max

    def normalize(self, value):
        res = self.param.normalize(value)
        if res is None:
            return None
        if self.min is not None:
            if res < self.min:
                return self.min
        if self.max is not None:
            if res > self.max:
                return self.max
        return res
