# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import warnings
import ema_workbench
from ema_workbench.connectors.netlogo import NetLogoModel
from ema_workbench import ema_logging, MultiprocessingEvaluator


NETLOGO_HOME = '/home/bcr/apps/netlogo'


def _build_executable_model(mediator, context):
	name = mediator.name
	model_dir_path = mediator.model['dir']
	model_file = mediator.model['name']
	netlogo_home = NETLOGO_HOME
	netlogo_version = 6
	gui = False

	uncertainties, outcomes = [], []

	for attr in context.resolution_model:
		if attr in mediator.feature_model.outcomes:
			outcomes.append(attr)
		else:
			uncertainties.append(attr)

	ema_model = NetLogoModel(name, model_dir_path, model_file, netlogo_home, netlogo_version, gui)
	ema_model.run_length = context.max_run_length
	ema_model.replications = context.num_repititions

	ema_model.uncertainties = uncertainties
	ema_model.outcomes = outcomes

	return ema_model


def synthesize(mediator, context):
	# ema_logging.log_to_stderr(ema_logging.INFO)
	ema_model = _build_executable_model(mediator, context)

	print(ema_model.outcomes)


	for o in ema_model.uncertainties:
		print(o)

	return None

	with MultiprocessingEvaluator(ema_model, 
								n_processes=context.num_processes, 
								maxtasksperchild=4) as evaluator:
		results = evaluator.perform_experiments(context.num_experiments)

	return results