# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import ema_workbench
from ema_workbench.connectors.netlogo import NetLogoModel
from ema_workbench import ema_logging, MultiprocessingEvaluator
import numpy as np
from .feature import CategoricalParameter


def _determine_parameters(mediator, context):
	resolution = set([fi for sft in context.resolution_model for fi in sft.collapse(ema_objects_only=False)])
	all_features = set(mediator.feature_model.collapse())
	difference = all_features.difference(resolution)
	move_difference_to_resolution = []

	for i, fi in enumerate(difference):
		if isinstance(fi, ema_workbench.CategoricalParameter):
			cats = {c.name: 0 for c in fi.categories}

			for j, fj in enumerate(resolution):
				if fj.name in cats:
					cats[fj.name] += 1

			cat_occurrences = sum(cats.values())
			if cat_occurrences != 1:
				move_difference_to_resolution.append(fi)
			else:
				for k, v in cats.items():
					if v == 1:
						fi.default = k
						break

	for f in move_difference_to_resolution:
		difference.remove(f)
		resolution.add(f)

	def is_outcome(_f):
		return isinstance(_f, ema_workbench.TimeSeriesOutcome) or \
			   isinstance(_f, ema_workbench.ArrayOutcome) or \
			   isinstance(_f, ema_workbench.ScalarOutcome)

	def is_ema_object(_f):
		return hasattr(_f, 'is_ema_object')

	constants = [f.to_constant() for f in difference if not is_outcome(f) and is_ema_object(f)]
	resolution_uncertainties = [f for f in resolution if not is_outcome(f) and is_ema_object(f)]
	resolution_outcomes = [f for f in resolution if is_outcome(f) and is_ema_object(f)]

	context.resolution_model = constants + resolution_uncertainties + resolution_outcomes
	return resolution_uncertainties, resolution_outcomes, constants


def _build_executable_model(mediator, context):
	# set the name of the EMAModel
	name = mediator.name

	# set the dir path for the model
	model_dir_path = mediator.model['dir']

	# set the name of the model
	model_file = mediator.model['name']

	# set the dir path where NetLogo is located
	netlogo_home = mediator.netlogo['dir']

	# set the NetLogo version
	netlogo_version = mediator.netlogo['version']

	# create the NetLogoModel object
	ema_model = NetLogoModel(name=name,
							 wd=model_dir_path,
							 model_file=model_file,
							 netlogo_home=netlogo_home,
							 netlogo_version=netlogo_version,
							 gui=False)

	# set the max run length of this model
	ema_model.run_length = context.max_run_length

	# set the number of replications
	ema_model.replications = context.num_replications

	uncertainties, outcomes, constants = _determine_parameters(mediator, context)
	ema_model.uncertainties = uncertainties
	ema_model.outcomes = outcomes
	ema_model.constants = constants
	return ema_model


def _get_exploratory_results(mediator, context):
	# turn logging on
	ema_logging.log_to_stderr(ema_logging.INFO)

	# create a EMA Model Object
	ema_model = _build_executable_model(mediator, context)

	with MultiprocessingEvaluator(ema_model,
								  n_processes=context.num_processes,
								  maxtasksperchild=4) as evaluator:
		# run model using EMA
		results = evaluator.perform_experiments(context.num_experiments)

		return results


def _normalize_experiments(context, exp_df, digitize=True):
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


def _shape_outcomes(outcomes_dict):
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


def _process_ema_results(context, results):
	# unpack the results as experiments and outcomes
	exp_df, out_dict = results

	# remove unnecessary columns
	exp_df = exp_df.drop(['policy', 'model', 'scenario'], axis=1)

	# get a normalized version of the experiments
	# returns a data frame
	experiments = _normalize_experiments(context, exp_df)

	# reshape the outcomes and take the mean over all repetitions
	# returns a dictionary:
	# <key>: (<# experiments>, <# outcomes>, <# repetition length>)
	outcomes = _shape_outcomes(out_dict)

	# return a dict containing both experiments and outcomes
	return {'experiments': experiments, 'outcomes': outcomes}


def _process_exploratory_results(context, results):
	# reshape and normalize the exploratory results data
	results = _process_ema_results(context=context, results=results)

	# separate experiments and outcomes
	experiments = results['experiments']

	# call the user-designed reward function to convert raw outcomes to reward
	outcomes = context.reward_function(*results['outcomes'])

	# create a new panda data frame to contain all the data
	data = experiments.copy()

	# insert the reward to the last column in the new data frame
	data.insert(len(data.columns), 'rho', outcomes)

	return data


def explore(mediator, context):
	results = _get_exploratory_results(mediator, context)
	processed_data = _process_exploratory_results(context, results)
	return processed_data

