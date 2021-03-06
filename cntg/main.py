#! /usr/bin/env python

from cn_generator import CN_Generator
from misc import NoGWError
from strategies.cost_strategy import CostStrategy
from strategies.cost_strategy_aggressive import CostStrategyAggressive
from strategies.growing_network import Growing_network
from strategies.pref_attachment import Pref_attachment
import configargparse

STRATEGIES = {
    'growing_network': Growing_network,
    'pref_attachment': Pref_attachment,
    'cost_sharing': CostStrategy,
    'cost_aggressive': CostStrategyAggressive
}


def parse_args():
    s_list = STRATEGIES.keys()
    parser = configargparse.get_argument_parser(default_config_files=['config.yml', 'experiment.yml'])
    parser.add_argument("-s", "--strategy",
                        help="a strategy to be used",
                        choices=s_list,
                        required=True)
    parser.add_argument("-d", "--dataset",
                        help="a data set from the available ones",
                        required=True)
    parser.add_argument("--max_dev",
                        help="maximum number of devices per node",
                        type=int, const=float('inf'), nargs='?',
                        default=float('inf'))
    parser.add_argument("-D", help="debug: print metrics at each iteration"
                                   " and save metrics in the './data' folder",
                        action='store_true')
    parser.add_argument("-P", "--processes", help="number of parallel processes",
                        default=1, type=int)
    parser.add_argument("-p", help="plot the graph using the browser",
                        dest='plot', action='store_true')
    parser.add_argument('-g', "--gateway", help="gateway number in [0,n] from gws.yml",
                        type=int, required=True)
    parser.add_argument('-n', "--max_size", help="number of nodes", type=int)
    parser.add_argument('-e', "--expansion", help="expansion range (in meters),"
                                                  " defaults to buildings at 30km", type=float,
                        default=30000)
    parser.add_argument('-r', "--seed", help="random seed,", type=int)
    parser.add_argument('-B', "--bandwidth", help="Accepts three arguments: bw frac min_n."
                                                  "Stop when a fraction of frac nodes has less than bw bandwidth. "
                                                  "Start measuring after min_n nodes (initially things may behave strangely "
                                                  "(in Mbps). Ex: '1 0 1' will stop when any node has less than 1Mbps "
                                                  "(in Mbps). Ex: '5 0.15 10' will stop when 15% of nodes has less than 5Mbps "
                                                  "but not before we have at least 10 nodes",
                        default="1 0 1")
    parser.add_argument('-R', "--restructure", help="restructure with edgeffect every r"
                                                    " rounds, adding l links. Accepts two arguments: r l", default=[])
    parser.add_argument('-V', "--viewshed_extra", help="Add at most v links extra link if"
                                                       "these are in the viewshed of the current one.", type=int,
                        default=0)
    parser.add_argument("-C", "--channel_width", help="802.11 channel width",
                        choices=[20, 40, 80, 160], default=20, type=int)
    parser.add_argument("--dsn", help="DSN to for the connection to PostGIS", required=True)
    parser.add_argument("--lidar_table", help="pointcloud table containing lidar/srtm data", required=True)
    parser.add_argument("--base_folder", help="Output base folder for the data", required=True)
    parser.add_argument("--cost_interface",
                        help="Cost interface used to calculate probabilities of node type in cost sharing mode", required=True)
    parser.add_argument("--show_level",
                        help="Show nodes type level. 0 = linked to newtork, 1 = also not interested, 2 also failed",
                        default=0, type=int)
    parser.add_argument("--eco_tb_name", help="Table name of economics data to use in CostInterface",
                        default=None)
    parser.add_argument("--sort_notlinked", help="Algorithm used to sort not linked node in CostAggressive strategy", default=3, type=int, choices=[0,1,2,3])
    parser.set_defaults(plot=False)
    args, unknown = parser.parse_known_args()
    return args, unknown


if __name__ == '__main__':
    args, unknown_args = parse_args()
    try:
        s = STRATEGIES.get(args.strategy)(args=args, unk_args=unknown_args)
    except NoGWError:
        print("Gateway Not provieded")
    else:
        s.main()
