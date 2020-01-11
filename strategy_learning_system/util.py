# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import ema_workbench
import numpy as np
import pandas as pd
import datetime
from .feature_model import CategoricalParameter


def clean_dir_path(dir):
	d = dir.replace('//', '/')
	return d


def datetime_str():
	dts = str(datetime.datetime.now())
	for e in ['.', ':', '-', ' ']:
		dts = dts.replace(e, '_')
	return dts


def normalize_experiments(context, exp_df, digitize=True):
	# for each column/uncertainty
	for column_name in exp_df:
		# grab the name of the param we are working on
		param = context[column_name]

		if param is None:
			continue

		if isinstance(param, CategoricalParameter):
			col = list(exp_df[column_name])
			col = [param.category_to_index[c] for c in col]
			exp_df[column_name] = col
		else:
			# take the min-max norm for each column
			# using the ranges defined in the experiments uncertainties
			col = exp_df[column_name]
			col = (col - param.lower_bound) / (param.upper_bound - param.lower_bound)

			# if digitize is true, then bin each of the attributes in exp_df
			if digitize:
				# if isinstance(context.bins, int):
				# 	bin_interval = 1.0 / context.bins
				# 	exp_df[column_name] = (exp_df[column_name] / bin_interval).astype(int)
				# else:
				# 	exp_df[column_name] = np.digitize(exp_df[column_name], context.bins, right=True)

				if isinstance(context.bins, int):
					n_bins = context.bins
				else:
					n_bins = len(context.bins)

				interval = 1 / n_bins

				# this "bins" elements in the array similar to np.digitize
				# except here we are setting the last bin as inclusive, inclusive
				for i in range(n_bins):
					lb = i * interval
					ub = (i + 1) * interval
					col[(lb <= col) & (col < ub)] = i

				exp_df[column_name] = col

	return exp_df


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

    # create a dictionary with all the outcomes:
    # <key>: (<# experiments>, <# outcomes>, <# repetition length>)

    return keys, outcomes


def process_ema_results(context, results):
	# unpack the results as experiments and outcomes
	exp_df, out_dict = results

	# remove unnecessary columns
	exp_df = exp_df.drop(['policy', 'model', 'scenario'], axis=1)

	# get a normalized version of the experiments
	# returns a data frame
	experiments = normalize_experiments(context, exp_df)

	# reshape the outcomes and take the mean over all repetitions
	# returns a dictionary:
	# <key>: (<# experiments>, <# outcomes>, <# repetition length>)
	outcomes = shape_outcomes(out_dict)

	# return a dict containing both experiments and outcomes
	return {'experiments': experiments, 'outcomes': outcomes}
