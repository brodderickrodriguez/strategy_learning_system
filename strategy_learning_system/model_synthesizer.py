# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import warnings
import ema_workbench
from ema_workbench.connectors.netlogo import NetLogoModel
from ema_workbench import ema_logging, MultiprocessingEvaluator


def _build_executable_model(mediator, context):
	# define lists for the model's uncertainties and outcomes
	uncertainties, outcomes = [], []

	# separate uncertainties and outcomes for EMA 
	for attr in context.resolution_model:
		if attr in mediator.feature_model.outcomes:
			outcomes.append(attr)
		else:
			uncertainties.append(attr)

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

	# set the model uncertainties and outcomes
	ema_model.uncertainties = uncertainties
	ema_model.outcomes = outcomes

	return ema_model


def synthesize(mediator, context):
	# turn logging on
	# ema_logging.log_to_stderr(ema_logging.INFO)

	# create a EMA Model Object
	ema_model = _build_executable_model(mediator, context)

	with MultiprocessingEvaluator(ema_model, 
								n_processes=context.num_processes, 
								maxtasksperchild=context.tasks_per_subchild) as evaluator:
		
		results = evaluator.perform_experiments(context.num_experiments)

	return results