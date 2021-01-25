"""emg_from_lfp

Usage:
  emg_from_lfp <emg_config_path>

Options:
  -h --help      show this
"""


import emg_from_lfp
from docopt import docopt
import yaml


if __name__ == "__main__":

    args = docopt(__doc__)

    # Load config
    emg_config_path = args["<emg_config_path>"]
    with open(emg_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Run main function
    emg_from_lfp.run(config)
