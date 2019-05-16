import math

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import numpy
from cost_model import IstatCostModel, TestCostModel, CostError, CostModel
from scipy.stats import norm
from geoalchemy2.shape import from_shape, to_shape

from misc import BuildingUtils


class CostInterfaceError(CostError):
    def __init__(self, msg):
        super().__init__(msg)


class CostInterface:
    parameters = {}

    def __init__(self, apply_model):
        self.cache = {}
        self.use_cache = True
        self.apply_model = apply_model

    def get(self, building):
        return self.apply_model(building, self.get_model(building), self.parameters)

    def get_model(self, building):
        raise NotImplementedError

    def get_cached(self, building):
        if not self.use_cache:
            return self.get(building)
        if building.gid not in self.cache.keys():
            self.cache[building.gid] = self.get(building)
        return self.cache[building.gid]

    def add_param(self, name, threshold_join, threshold_super):
        self.parameters[name] = (threshold_join, threshold_super)


class IstatCostInterface(CostInterface):

    def __init__(self, DSN, bi):
        super().__init__(AppliedVolumeModel)
        self.engine = create_engine(DSN, client_encoding='utf8', echo=False)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.bi = bi
        self.parameters["people"] = PolyCostParam([(1, 0), (5, 1)])
        # self.PARAMS["test"] = (1, 4)

    def load_model_by_area(self, area):
        # get all the istat models intersecting the area
        result = self.session.query(IstatCostModel).filter(IstatCostModel.geom.ST_Intersects(area)).first()
        if result is not None:
            buildings = result.load_buildings(self.bi)
            for building in buildings:
                self.cache[building.gid] = self.apply_model(building, result, self.parameters)
        return result

    def get_model(self, building):
        return self.load_model_by_area(building.geom)


class CostParam:
    def __init__(self, weight=1):
        self.weight = weight

    def normalize(self, value):
        raise NotImplementedError


class PolyCostParam(CostParam):
    def normalize(self, value):
        return numpy.polyval(self.polynomial, value)

    def __init__(self, points, degree=-1, weight=1):
        super().__init__(weight)
        x, y = [], []
        for point in points:
            x.append(point[0])
            y.append(point[1])
        if degree < 0:
            degree = len(points) - 1
        self.polynomial = numpy.polyfit(x, y, degree)




class AppliedCostModel:
    def __init__(self, building, model, params):
        self.model = model
        self.building = building
        self.params = params
        self.properties = {**model.get_properties()}
        self.probabilities = None

    def applied_metrics(self):
        return self.model.get_metrics()

    def get_mean_std_percentage(self):
        metrics = self.applied_metrics()
        print(metrics)
        normalized, weights = [], []
        for name, param in self.params.items():
            try:
                metric = metrics[name]
            except IndexError:
                raise CostError("Unknown model metric: {}".format(name))
            prc = param.normalize(metric)
            wei = param.weight
            self.properties["Param " + name] = "{:.2f} - {:.2f} - {:.2f}".format(metric, prc, wei)
            normalized.append(prc)
            weights.append(wei)
        print(normalized)
        return numpy.average(normalized, weights=weights), (numpy.sqrt(numpy.cov(normalized, aweights=weights))
                                                            if len(normalized) > 1 else 0.2)

    def get_probabilities(self):
        if self.probabilities is None:
            avg, std = self.get_mean_std_percentage()
            self.properties['Average interest'] = avg
            self.properties['Std interest'] = std
            # print(mean, std)
            # print(self.model.get_std_dev())
            std = (2.2 / (1 + math.exp(-1 * (math.sqrt(self.model.get_std_dev() + std))))) - 1
            # print(std)
            ths = norm(avg, std).cdf([0, 1])
            p1 = ths[0]
            p2 = ths[1] - ths[0]
            p3 = 1 - ths[1]
            self.properties['prob'] = "{:.2f}% - {:.2f}% - {:.2f}%".format(p1 * 100, p2 * 100, p3 * 100)
            self.probabilities = ths
        return self.probabilities


class AppliedVolumeModel(AppliedCostModel):
    def applied_metrics(self):
        metrics = self.model.get_metrics()
        volume = BuildingUtils.get_building_volume(self.building)
        self.properties["Volume"] = volume
        total_volume = self.model.get_buildings_volume()
        return {key: (volume / total_volume * val) for key, val in metrics.items()}
