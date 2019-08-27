# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019


# TODO: I/O

import ema_workbench
import numpy as np


def load_results(data_dir):
    # use EMA's built-in load_results function to unzip and load experiment results
    exp_df, out_dict = ema_workbench.load_results(data_dir)

    # remove unnecessary columns
    exp_df = exp_df.drop(['policy', 'model', 'scenario'], axis=1)

    # return tuple containing the experiments and outcomes 
    return exp_df, out_dict
