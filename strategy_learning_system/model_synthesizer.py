# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import warnings
import ema_workbench
from ema_workbench.connectors.netlogo import NetLogoModel
from ema_workbench import ema_logging, MultiprocessingEvaluator


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

	def is_outcome(f):
		return isinstance(f, ema_workbench.TimeSeriesOutcome) or \
			   isinstance(f, ema_workbench.ArrayOutcome) or \
			   isinstance(f, ema_workbench.ScalarOutcome)

	def is_ema_object(f):
		return hasattr(f, 'is_ema_object')

	constants = [f.to_constant() for f in difference if not is_outcome(f) and is_ema_object(f)]
	resolution_uncertainties = [f for f in resolution if not is_outcome(f) and is_ema_object(f)]
	resolution_outcomes = [f for f in resolution if is_outcome(f) and is_ema_object(f)]

	context.all_parameters = constants + resolution_uncertainties + resolution_outcomes

	# context.resolution_model = constants + resolution_uncertainties + resolution_outcomes
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

	# turn GUI off
	gui = False

	# create the NetLogoModel object
	ema_model = NetLogoModel(name=name, 
							wd=model_dir_path, 
							model_file=model_file, 
							netlogo_home=netlogo_home, 
							netlogo_version=netlogo_version, 
							gui=gui)
	
	# set the max run length of this model
	ema_model.run_length = context.max_run_length

	# set the number of replications
	ema_model.replications = context.num_replications

	uncertainties, outcomes, constants = _determine_parameters(mediator, context)
	ema_model.uncertainties = uncertainties
	ema_model.outcomes = outcomes
	ema_model.constants = constants
	return ema_model


def synthesize(mediator, context):
	# turn logging on
	ema_logging.log_to_stderr(ema_logging.INFO)

	# create a EMA Model Object
	ema_model = _build_executable_model(mediator, context)

	with MultiprocessingEvaluator(ema_model, 
								n_processes=context.num_processes, 
								maxtasksperchild=context.tasks_per_subchild) as evaluator:
		
		# run model using EMA
		results = evaluator.perform_experiments(context.num_experiments)

		return results
