# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import ema_workbench
import numpy as np
import pandas as pd
import datetime


def clean_dir_path(dir):
	d = dir.replace('//', '/')
	return d


def datetime_str():
	dts = str(datetime.datetime.now())
	for e in ['.', ':', '-', ' ']:
		dts = dts.replace(e, '_')
	return dts


def normalize_experiments(context, exp_df):
    # a list containing the ordered uncertainties 
    keys = []

    # for each column/uncertainty 
    for column_name in exp_df:
        # grab the name of the param we are working on
        param = context[column_name]

        # add the param name to the ordered keys list
        keys.append(column_name)

        # take the min-max norm for each column
        # using the ranges defined in the experiments uncertainties
        exp_df[column_name] = (exp_df[column_name] - param.lower_bound) / (param.upper_bound - param.lower_bound)
    
    # return a tuple containing the ndarray experiments and experiments' keys
    return exp_df.to_numpy(), keys


def shape_outcomes(outcomes_dict):
    # create a list containing the names of all the outcomes
    keys = list(outcomes_dict.keys())

    # create and fill ndarray outcomes (4 dims)
    # shape here is (<# outcomes>, <# experiments>, <# repetitions>, <# repetition length>)
    outcomes = np.array([outcomes_dict[key] for key in keys])

    # swap axis 0 and 1 so we get experiments as axis 0 (4 dims)
    # shape here is (<# experiments>, <# outcomes>, <# repetitions>, <# repetition length>)
    outcomes = np.swapaxes(outcomes, 0, 1)

    # take the mean over all repetitions (3 dims)
    # shape here is (<# experiments>, <# outcomes>, <# repetition length>)
    outcomes = np.nanmean(outcomes, axis=2)

    # return tuple containing outcomes and outcomes' keys
    return outcomes, keys


def process_ema_results(context, results):
	# unpack the results as experiments and outcomes
	exp_df, out_dict = results

	# remove unnecessary columns
	exp_df = exp_df.drop(['policy', 'model', 'scenario'], axis=1)

	# get a normalized version of the experiments
	# returns a tuple: (<experiments>, <uncertainties' names>)
	experiments = normalize_experiments(context, exp_df)

	# reshape the outcomes and take the mean over all repetitions
	# returns a tuple: (<outcomes>, <outcomes' names>)
	outcomes = shape_outcomes(out_dict)

	# return a dict containing both experiments and outcomes
	return {'experiments': experiments, 'outcomes': outcomes}
