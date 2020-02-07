"""EMGfromLFP

Usage:
  EMGfromLFP <EMG_config_path>

Options:
  -h --help      show this
"""


import EMGfromLFP
from docopt import docopt
import yaml


if __name__ == '__main__':

    args = docopt(__doc__)

    # Load config
    EMG_config_path = args['<EMG_config_path>']
    with open(EMG_config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Run main function
    EMGfromLFP.run(config)
