from enum import Enum

import math

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import numpy
from cost_model import IstatCostModel, CostError, MetricType, ConstCostModel
from scipy.stats import norm
from libterrain.building import DataUnavailable

from misc import BuildingUtils
from node import CostChoice


class CostInterfaceError(CostError):
    """Raised from cost-related function errors"""
    pass


class CostInterface:
    parameters = {}

    def __init__(self, apply_model):
        self.cache = {}
        self.use_cache = True
        self.apply_model = apply_model

    def get(self, building):
        a = self.apply_model(building, self.get_model(building), self.parameters)
        return a

    def get_model(self, building):
        raise NotImplementedError

    def get_cached(self, building):
        if not self.use_cache:
            return self.get(building)
        if building.gid not in self.cache.keys():
            self.cache[building.gid] = self.get(building)
        return self.cache[building.gid]


class IstatCostInterface(CostInterface):

    def __init__(self, DSN, bi):
        super().__init__(AppliedVolumeModel)
        self.engine = create_engine(DSN, client_encoding='utf8', echo=False)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.bi = bi
        self.parameters["people"] = PolyCostParam([(0.5, 0), (16, 1)])
        self.parameters["avg_age"] = SegmentCostParam(
            [(0, -1), (10, 0), (17, 0.6), (20, 0.7), (23, 1), (33, 1.5), (40, 1), (50, 0.9), (60, 0.7), (70, 0.5),
             (100, 0.2)], 0.5)
        # self.PARAMS["test"] = (1, 4)

    def load_model_by_area(self, area):
        # get all the istat models intersecting the area
        result = self.session.query(IstatCostModel).filter(IstatCostModel.geom.ST_Intersects(area)).first()
        if result is not None:
            print("Loading buildings of istat area: ", result.gid)
            buildings = result.load_buildings(self.bi)
            for building in buildings:
                try:
                    self.cache[building.gid] = self.apply_model(building, result, self.parameters)
                except CostError:
                    pass
        return result

    def get_model(self, building):
        return self.load_model_by_area(building.geom)


class ConstCostInterface(CostInterface):

    def __init__(self, cost_choice):
        super().__init__(AppliedConstModel)
        if cost_choice is CostChoice.SUPER_NODE:
            self.ths = (0, 0)
        elif cost_choice is CostChoice.LEAF_NODE:
            self.ths = (0, 1)
        else:
            self.ths = (1, 1)

    def get_model(self, building):
        return ConstCostModel({})

    def get(self, building):
        return AppliedConstModel(building, self.ths)


class CostParam:
    def __init__(self, weight=1):
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


class AppliedCostModel:
    @staticmethod
    def _weighted_avg_std(values, weights):
        if len(values) == 0 or len(values) != len(weights):
            raise ValueError("Invalid number of weights" if values else "Empty value list")
        average = numpy.average(values, weights=weights)
        variance = numpy.average([(value - average) ** 2 for value in values], weights=weights)
        return average, math.sqrt(variance)

    @staticmethod
    def normalize_std_dev(x):
        return (3 / (1 + math.exp(-1 / 2 * (2 * x)))) - 1.4

    def __init__(self, building, model, params):
        self.model = model
        self.building = building
        self.params = params
        self.properties = {**model.get_properties()}
        self.probabilities = None
        self.weight = 0 if model is None else model.get_weight()

    def applied_metrics(self):
        return {key: val[0] for key, val in self.model.get_metrics_cached().items()}

    def get_mean_std_percentage(self):
        metrics = self.applied_metrics()
        normalized, weights = [], []

        for name, param in self.params.items():
            metric = metrics[name]
            prc = param.normalize(metric)
            wei = param.weight
            self.properties["Param " + name] = "Val:{:.2f} - Norm:{:.2f} - Wei:{:.2f}".format(metric, prc, wei)
            normalized.append(prc)
            weights.append(wei)
        avg = numpy.average(normalized, weights=weights)
        total_distance = 0
        distances = []
        for v in normalized:
            d = abs(v - avg)
            distances.append(d)
            total_distance += d
        for i in range(len(weights)):
            weights[i] *= (1 - distances[i] / total_distance)
        return self._weighted_avg_std(normalized, weights)

    def get_probabilities(self):
        if self.probabilities is None:
            avg, std = self.get_mean_std_percentage()
            std = (self.model.get_std_dev() + std)/2
            self.properties['Average interest'] = avg
            self.properties['Std interest'] = std
            # print(mean, std)
            # print(self.model.get_std_dev())
            std = self.normalize_std_dev(std)
            self.properties['Normalized std'] = std
            # print(std)
            self.probabilities = norm(avg, std).cdf([0, 1])
        return self.probabilities


class AppliedVolumeModel(AppliedCostModel):
    def __init__(self, building, model, params):
        super().__init__(building, model, params)
        try:
            volume = BuildingUtils.get_building_volume(self.building)
            if volume <= 0:
                raise DataUnavailable('Invalid volume: %s' % volume)
            total_volume = self.model.get_buildings_volume()
            if total_volume < volume:
                raise DataUnavailable('Invalid total volume: %s < %s' % (total_volume, volume))
        except DataUnavailable as e:
            raise CostInterfaceError('Failed to calculate volumes: %s' % e) from e
        self.properties["Volume"] = volume
        self.factor = volume / total_volume
        self.weight *= self.factor

    def applied_metrics(self):
        metrics = self.model.get_metrics_cached()
        met = {}
        for key, val in metrics.items():
            value = val[0]
            type = val[1]
            if value is None:
                met[key] = None
                continue
            if type is MetricType.UNIQUE:
                value = value * self.factor
            met[key] = value
        return met


class AppliedConstModel(AppliedCostModel):
    def __init__(self, building, thresholds):
        super().__init__(building, ConstCostModel({}), {})
        self.ths = thresholds

    def get_probabilities(self):
        return self.ths
