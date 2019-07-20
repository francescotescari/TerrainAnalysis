import random

import configargparse
import yaml
from geoalchemy2.shape import to_shape, from_shape
from libterrain import OSMInterface, CTRInterface
from shapely.geometry import Point
from shapely.ops import cascaded_union
from sqlalchemy import create_engine, Column, Integer, BigInteger, MetaData, Table, Float, ForeignKey, inspect, case, \
    text, func, distinct, and_, ARRAY, exists, cast, literal, Index, literal_column, not_
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, mapper
from sqlalchemy.testing import in_

from misc import BuildingUtils, Susceptible_Buffer, NoGWError


class AbstractPopulator():

    def __init__(self, bi, tbname, args=None, unkn_args=None):
        pass

    def populate(self, geom=None):
        raise NotImplementedError


class IstatPopulator(AbstractPopulator):

    def __init__(self, bi, tbname, args=None, unkn_args=None):
        super().__init__(bi, tbname, args, unkn_args)
        self.engine = bi.engine
        self.meta = bi.meta
        self.bi = bi
        self.tb_name = tbname
        self.session = None
        self.metrics = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p12", "p11", "p13", "p14", "p15",
                        "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p25", "p26", "p27", "p28",
                        "p29", "p30", "p31", "p32", "p33", "p34", "p35", "p36", "p37", "p38", "p39", "p40", "p41",
                        "p42", "p43", "p44", "p45", "p46", "p47", "p48", "p49", "p50", "p51", "p52", "p53", "p54",
                        "p55", "p56", "p57", "p58", "p59", "p60", "p61", "p62", "p64", "p65", "p66", "p128", "p129",
                        "p130", "p131", "p132", "p135", "p136", "p137", "p138", "p139", "p140", "st1", "st2", "st3",
                        "st4", "st5", "st6", "st7", "st8", "st9", "st10", "st11", "st12", "st13", "st14", "st15", "a2",
                        "a3", "a5", "a6", "a7", "a44", "a46", "a47", "a48", "pf1", "pf2", "pf3", "pf4", "pf5", "pf6",
                        "pf7", "pf8", "pf9", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e12",
                        "e13", "e14", "e15", "e16", "e17", "e18", "e19", "e20", "e21", "e22", "e23", "e24", "e25",
                        "e26", "e27", "e28", "e29", "e30", "e31"]
        self.tolerance = 1-args.tolerance
        self.drop = args.drop

    def new_session(self):
        session = sessionmaker(bind=self.engine)
        return session()

    def populate(self, geom=None):

        bi = self.bi
        B = bi.building_class
        eco_table = Table(self.tb_name, self.meta, Column('gid', Integer, ForeignKey(B.gid), primary_key=True),
                          Column('std_dev', Float), *[Column(m, Float) for m in self.metrics])

        self.meta.reflect(bind=self.engine, only=['istat', 'basi_istat'], views=True)
        istat_table = self.meta.tables['istat']

        class IstatMapper:
            pass

        class EcoMapper:
            pass

        mapper(EcoMapper, eco_table)
        mapper(IstatMapper, istat_table, primary_key=istat_table.columns['gid'])
        rel_tbname = self.tb_name + "_rel"
        rel_table = Table(rel_tbname, self.meta, Column('gid', Integer, ForeignKey(EcoMapper.gid)),
                          Column('istat_sez2011', BigInteger, ForeignKey("basi_istat.sez2011")))
        if geom is None:
            shape = None
        else:
            session = self.new_session()
            istat_areas = session.query(IstatMapper).filter(IstatMapper.geom.ST_Transform(4326).ST_Intersects(geom)).all()
            shape = from_shape(cascaded_union([to_shape(o.geom) for o in istat_areas]), 4326)
            session.close()
        create_table = True
        create_table_rel = True
        if geom is None or self.drop:
            if rel_table.exists(bind=self.engine):
                print("Dropping old relationship table: %s" % rel_tbname)
                rel_table.drop(bind=self.engine)
            if eco_table.exists(bind=self.engine):
                print("Dropping old records table: %s" % self.tb_name)
                eco_table.drop(bind=self.engine)
        else:
            create_table = not eco_table.exists(bind=self.engine)
            create_table_rel = not rel_table.exists(bind=self.engine)

        if create_table:
            print("Creating records table: %s" % self.tb_name)
            eco_table.create(bind=self.engine)
        if create_table_rel:
            print("Creating relationship table: %s" % rel_tbname)
            rel_table.create(bind=self.engine)

        self.session = self.new_session()
        I = IstatMapper
        H = bi.height_class
        if B is None or H is None:
            raise Exception("Missing mapper info")
        domain = self.session.query(B)
        if shape is not None:
            domain = domain.filter(shape.ST_Intersects(B.geom))
        domain = domain.filter(B.gid.notin_(self.session.query(EcoMapper.gid).cte('present'))).cte('domain').c
        sub = self.session.query(domain.gid.label('bgid'), I.sez2011, domain.geom.label('bgeom'), I.geom.label('igeom'), domain.geom.ST_Transform(3003).ST_Area().label('barea'), func.count().over(partition_by=domain.gid).label('num_sez'), func.count().over(partition_by=I.sez2011).label('bcount'), *[getattr(I, m) for m in self.metrics])
        sub = sub.join(I, domain.geom.ST_Intersects(I.geom)).subquery('sub')
        sub = self.session.query(sub, case([(sub.c.num_sez > 1, sub.c.bgeom.ST_Intersection(sub.c.igeom).ST_Area())], else_=1).label('int_area'), (sub.c.barea * (H.dsm_avg - H.dtm_avg)).label('volume')).join(H, and_(H.gid == sub.c.bgid, H.dsm_avg > H.dtm_avg)).subquery('sub')
        sub = self.session.query(sub, (sub.c.int_area / func.sum(sub.c.int_area).over(partition_by=sub.c.bgid)).label('a_ratio')).subquery('sub')
        sub = self.session.query(sub, (sub.c.volume * sub.c.a_ratio).label('d_volume')).subquery('sub')
        sub = self.session.query(sub, (cast(func.count().over(partition_by=sub.c.sez2011), Float) / cast(sub.c.bcount, Float)).label('good'), (func.stddev(sub.c.d_volume).over(partition_by=sub.c.sez2011)/func.avg(sub.c.d_volume).over(partition_by=sub.c.sez2011)).label('std_dev'), sub.c.bcount, (sub.c.d_volume / func.sum(sub.c.d_volume).over(partition_by=sub.c.sez2011)).label( 'e_ratio')).subquery('sub')
        query = self.session.query(sub.c.bgid.label('gid'), sub.c.sez2011.label('iid'), sub.c.e_ratio.label('ratio'), sub.c.std_dev, sub.c.a_ratio, *[getattr(sub.c, m).label(m) for m in self.metrics]).filter( and_(sub.c.e_ratio > 0, sub.c.e_ratio <= 1)).filter(sub.c.good >= self.tolerance)
        tmp_table = query.cte('tmp_table')
        data_select = self.session.query(tmp_table.c.gid, (func.sum(tmp_table.c.std_dev*tmp_table.c.a_ratio)/func.sum(tmp_table.c.a_ratio)),*[func.sum(tmp_table.c.ratio * getattr(tmp_table.c, m)).label(m) for m in self.metrics]).group_by(tmp_table.c.gid)
        rel_select = self.session.query(tmp_table.c.gid, tmp_table.c.iid)
        ins1 = eco_table.insert().from_select(['gid', 'std_dev', *self.metrics], data_select).returning(literal(1).label('res'))
        ins2 = rel_table.insert().from_select(['gid', 'istat_sez2011'], rel_select).returning(literal(1).label('res'))
        sub1 = self.session.query(func.count(ins1.cte('ins1').c.res)).subquery('sub1')
        sub2 = self.session.query(func.count(ins2.cte('ins2').c.res)).subquery('sub2')
        query = self.session.query(sub1, sub2)
        res = query.all()
        self.session.commit()
        print("Inserted %d new records, %d relationships" % (res[0][0], res[0][1]))

        self.session.close()


BUILDING_INTERFACES = {
    'osm': OSMInterface,
    'ctr': CTRInterface
}

POPULATORS = {
    'istat': IstatPopulator,
}


def parse_args():
    parser = configargparse.get_argument_parser(default_config_files=['config.yml', 'experiment.yml'])
    parser.add_argument("-d", "--dataset",
                        help="a data set from the available ones")
    parser.add_argument('-g', "--gateway", help="gateway number in [0,n] from gws.yml",
                        type=int)
    parser.add_argument('-r', "--radius", help="max radius of scan area from the gateway", type=float)
    parser.add_argument("--dsn", help="DSN to for the connection to PostGIS", required=True)
    parser.add_argument("-bi", "--building_interface",
                        help="Choose the building interface to be linked to building data",
                        choices=BUILDING_INTERFACES.keys(), default='osm')
    parser.add_argument("-etb", "--eco_tb_name", help="Output table name",
                        default=None)
    parser.add_argument("--populator", help="Populator name", default="istat", choices=POPULATORS.keys())
    parser.add_argument("--tolerance", help="Max percentage of faulty data", default=0.15, type=float)
    parser.add_argument("--drop", help="Drop existing data first",  dest='drop', action='store_true')
    return parser.parse_known_args()


def die(msg):
    print(msg)
    exit(1)


if __name__ == '__main__':
    args, unknown_args = parse_args()
    bi = BUILDING_INTERFACES[args.building_interface](args.dsn)
    geom = None
    radius = None if (args.radius is None or args.radius < 0) else args.radius
    if args.dataset is not None and args.gateway is not None and radius is not None:
        if radius != 0:
            try:
                with open("gws.yml", "r") as gwf:
                    gwd = yaml.load(gwf, Loader=yaml.FullLoader)
                    try:
                        position = gwd['gws'][args.dataset][args.gateway]
                    except IndexError:
                        raise NoGWError("Index %d is out of range" % args.gateway)
                    except KeyError:
                        raise NoGWError("Dataset %s is not in gw file" % args.dataset)
                    gw_pos = Point(float(position[1]), float(position[0]))
                    sb = Susceptible_Buffer()
                    sb.set_shape([gw_pos])
                    geom = from_shape(sb.get_buffer(radius), srid=4326)
            except Exception as e:
                die("Failed to get gateway: %r" % e)
    elif args.dataset is not None:
        try:
            area = bi.get_province_area(args.dataset)
            if area is None:
                raise Exception("None area obj")
            geom = from_shape(area, 4326)
        except Exception as e:
            die("Failed to get province area: %r" % e)
    try:
        populator = POPULATORS[args.populator]
    except KeyError:
        die('Unknown data populator: %s' % args.populator)
    table_name = 'eco_data_' + bi.building_class.__tablename__ if args.eco_tb_name is None else args.eco_tb_name
    populator(bi, table_name, args, unknown_args).populate(geom)
