from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer
from geoalchemy2 import Geometry
from geoalchemy2.shape import to_shape
from numpy import std

from misc import BuildingUtils

Base = declarative_base()


class CostError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class CostModel(Base):
    """Cost model object indicates the interest for a building/area/whatever
    Core object to implement and use in cost interface
    """
    __abstract__ = True

    def get_metrics(self):
        raise NotImplementedError

    def get_properties(self):
        return {}

    def get_std_dev(self):
        return 0.2



class TestCostModel(CostModel):
    __abstract__ = True

    def __init__(self, gid):
        self.gid = gid

    def get_metrics(self):
        return {
            'gid_modulus': self.gid % 3
        }


class IstatCostModel(CostModel):
    __tablename__ = 'istat'
    gid = Column(Integer, primary_key=True)
    pp_number = Column("p1", Integer)
    pp_year0_5 = Column("p14", Integer)
    pp_year5_10 = Column("p15", Integer)
    pp_year10_15 = Column("p16", Integer)
    pp_year15_20 = Column("p17", Integer)
    pp_year20_25 = Column("p18", Integer)
    pp_year25_30 = Column("p19", Integer)
    pp_year30_35 = Column("p20", Integer)
    pp_year35_40 = Column("p21", Integer)
    pp_year40_45 = Column("p22", Integer)
    pp_year45_50 = Column("p23", Integer)
    pp_year50_55 = Column("p24", Integer)
    pp_year55_60 = Column("p25", Integer)
    pp_year60_65 = Column("p26", Integer)
    pp_year65_70 = Column("p27", Integer)
    pp_year70_75 = Column("p28", Integer)
    pp_year75_200 = Column("p29", Integer)
    pp_university = Column("p47", Integer)
    pp_high_school = Column("p48", Integer)
    pp_mid_school = Column("p49", Integer)
    pp_elementary_school = Column("p50", Integer)
    pp_literate = Column("p51", Integer)
    pp_illiterate = Column("p52", Integer)
    pp_job_seeker = Column("p60", Integer)
    pp_employed = Column("p61", Integer)
    pp_unemployed = Column("p128", Integer)
    pp_household = Column("p130", Integer)
    pp_student = Column("p131", Integer)

    geom = Column(Geometry('POLYGON'))



    def __init__(self):
        self.buildings = None
        self.buildings_volume = None
        self.buildings_std = None

    def load_buildings(self, building_interface):
        self.buildings = building_interface.get_buildings(self.shape())
        return self.buildings

    def shape(self):
        return to_shape(self.geom)

    def require_buildings(self):
        if not self.buildings:
            raise CostError("Missing buildings information data, did you load the buildings data first?")
        return self.buildings

    def get_buildings_volume(self):
        if getattr(self, "buildings_volume", None) is None:
            buildings = self.require_buildings()
            self.buildings_volume = 0
            for building in buildings:
                self.buildings_volume += BuildingUtils.get_building_volume(building)
        return self.buildings_volume

    def get_buildings_number(self):
        return len(self.require_buildings())

    def get_metrics(self):
        return {
            'people': self.pp_number,
            'employed': self.pp_employed
        }

    def get_properties(self):
        return {
            'Istat gid': self.gid
        }

    def get_std_dev(self):
        if getattr(self, "buildings_std", None) is None:
            self.buildings_std = std([BuildingUtils.get_building_volume(b) for b in self.require_buildings()])*1000000
        return self.buildings_std
