import math
import numpy
from scipy.stats import norm, truncnorm


class CostError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class AppliedModel:
    mitigate_outlier = (0.5, 2)  # [0] = distance_{0} added to each distance, [1] = exponent of distance
    limit_std = (0.0001, 1.55, 1.8)  # limiting function for std: (min, max, speed to limit)
    prop_domain = (-1.3, 2.3)  # interest domain limits (min, max)
    flat_uncertainty = 0.1

    @staticmethod
    def _weighted_avg_std(values, weights):
        if len(values) == 0 or len(values) != len(weights):
            raise ValueError("Invalid number of weights" if len(values) != 0 else "Empty value list")
        average = numpy.average(values, weights=weights)
        variance = numpy.average([(value - average) ** 2 for value in values], weights=weights)
        return average, math.sqrt(variance)

    def limit_std_dev(self, x):
        return (self.std_norm_a / (1 + math.exp(-1* self.std_norm_c *x))) + self.std_norm_b

    def __init__(self, building, params, metrics=None, properties=None, model_std=0, weight=1):
        self.building = building
        self.params = params
        self.probabilities = None
        self.properties = {} if properties is None else {**properties}
        self.metrics = {} if metrics is None else {**metrics}
        self.model_std = model_std
        self.weight = weight
        if self.limit_std is not None:
            self.std_norm_a = 2 * (self.limit_std[1] - self.limit_std[0])
            self.std_norm_b = self.limit_std[1] - self.std_norm_a
            self.std_norm_c = self.limit_std[2]

    def _get_adjusted_data(self, values, weights):
        new_weights = weights
        if self.mitigate_outlier is not None :
            avg = numpy.average(values, weights=weights)
            new_weights = []
            for i in range(len(weights)):
                new_weights.append(weights[i] / (abs(values[i] - avg)**self.mitigate_outlier[1] + self.mitigate_outlier[0]))
        return values, new_weights

    def _get_mean_std_percentage(self):
        normalized, weights = [], []
        for name, param in self.params.items():
            metric = self.metrics[name]
            prc = param.normalize(metric)
            if prc is None:
                continue
            wei = param.weight
            self.properties["Param %s" % name] = "Val:{:.2f} - Norm:{:.2f} - Wei:{:.2f}".format(metric, prc, wei)
            normalized.append(prc)
            weights.append(wei)
        if len(normalized) == 0:
            raise CostError("No valid normalized parameter found")
        values, weights = self._get_adjusted_data(normalized, weights)
        return self._weighted_avg_std(values, weights)

    def get_probabilities(self):
        if self.probabilities is None:
            avg, std = self._get_mean_std_percentage()
            if self.model_std is not None:
                std = (self.model_std + std) / 2
            if self.flat_uncertainty is not None:
                std += self.flat_uncertainty
            self.properties['Average interest'] = avg
            self.properties['Std interest'] = std
            if self.limit_std is not None:
                std = self.limit_std_dev(std)
                self.properties['Limited std'] = std
            if self.prop_domain is not None:
                a, b = (self.prop_domain[0] - avg) / std, (self.prop_domain[1] - avg) / std
                self.probabilities = truncnorm.cdf([0, 1], a=a, b=b, loc=avg, scale=std)
            else:
                self.probabilities = norm(avg, std).cdf([0, 1])
        return self.probabilities


class CostModel:
    """Cost model object indicates the interest for a building/area/whatever
    Core object to implement and use in cost interface
    """
    cache = True

    def __init__(self, mapped_entry=None):
        self.metrics_cached = None
        self.properties = {}
        self._init_mapped_entry(mapped_entry)

    def _init_mapped_entry(self, mapped_entry):
        pass

    def get_weight(self):
        raise NotImplementedError

    def get_metrics(self):
        raise NotImplementedError

    def get_metrics_cached(self):
        if getattr(self, 'metrics_cached', None) is None or not self.cache:
            self.metrics_cached = self.get_metrics()
        return self.metrics_cached

    def get_properties(self):
        return self.properties

    def get_std_dev(self):
        return 0.1


class IstatCostModel(CostModel):

    def __init__(self, istat_db_entry):
        super().__init__(istat_db_entry)
        self.std_dev = None

    def _init_mapped_entry(self, mapped_entry):
        self.mapped_entry = mapped_entry
        self.pp_number = mapped_entry.p1
        self.pp_year0_5 = mapped_entry.p14
        self.pp_year5_10 = mapped_entry.p15
        self.pp_year10_15 = mapped_entry.p16
        self.pp_year15_20 = mapped_entry.p17
        self.pp_year20_25 = mapped_entry.p18
        self.pp_year25_30 = mapped_entry.p19
        self.pp_year30_35 = mapped_entry.p20
        self.pp_year35_40 = mapped_entry.p21
        self.pp_year40_45 = mapped_entry.p22
        self.pp_year45_50 = mapped_entry.p23
        self.pp_year50_55 = mapped_entry.p24
        self.pp_year55_60 = mapped_entry.p25
        self.pp_year60_65 = mapped_entry.p26
        self.pp_year65_70 = mapped_entry.p27
        self.pp_year70_75 = mapped_entry.p28
        self.pp_year75_200 = mapped_entry.p29
        self.pp_university = mapped_entry.p47
        self.pp_high_school = mapped_entry.p48
        self.pp_mid_school = mapped_entry.p49
        self.pp_elementary_school = mapped_entry.p50
        self.pp_literate = mapped_entry.p51
        self.pp_illiterate = mapped_entry.p52
        self.pp_job_seeker = mapped_entry.p60
        self.pp_employed = mapped_entry.p61
        self.pp_unemployed = mapped_entry.p128
        self.pp_household = mapped_entry.p130
        self.pp_student = mapped_entry.p131
        self.std_dev = mapped_entry.std_dev

    def get_weight(self):
        return self.pp_number

    def get_metrics(self):
        return {
            'people': self.pp_number,
            'employed': self.pp_employed,
            'avg_age': self.average_age()
        }

    def get_std_dev(self):
        return self.std_dev

    def average_age(self):
        age = 2.5 * self.pp_year0_5 + 7.5 * self.pp_year5_10 + 12.5 * self.pp_year10_15 + 17.5 * self.pp_year15_20 + \
              22.5 * self.pp_year20_25 + 27.5 * self.pp_year25_30 + 32.5 * self.pp_year30_35 + 37.5 * self.pp_year35_40 + \
              42.5 * self.pp_year40_45 + 47.5 * self.pp_year45_50 + 52.5 * self.pp_year50_55 + 57.5 * self.pp_year55_60 + \
              62.5 * self.pp_year60_65 + 67.5 * self.pp_year65_70 + 72.5 * self.pp_year70_75 + 80 * self.pp_year75_200
        return (age / self.pp_number) if self.pp_number > 0 else 0
