# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import warnings
import ema_workbench
from ema_workbench.connectors.netlogo import NetLogoModel
from ema_workbench import ema_logging, MultiprocessingEvaluator


NETLOGO_HOME = '/home/bcr/apps/netlogo'


def _build_executable_model(model_info, resolution_model, max_run_length, num_repititions):
	ema_model = NetLogoModel(model_info['inter_name'], 
							wd=model_info['dir'], 
							model_file=model_info['name'], 
							netlogo_home=NETLOGO_HOME, 
							netlogo_version=6, 
							gui=False)

	ema_model.max_run_length = max_run_length
	ema_model.replications = num_repititions

	uncertainties, outcomes = resolution_model
	ema_model.uncertainties = uncertainties
	ema_model.outcomes = outcomes

	return ema_model


def execute(model_info, resolution_model, num_experiments, max_run_length, num_repititions, num_processes):
	# ema_logging.log_to_stderr(ema_logging.INFO)
	ema_model = _build_executable_model(model_info, resolution_model, max_run_length, num_repititions)

	with MultiprocessingEvaluator(ema_model, n_processes=num_processes, maxtasksperchild=4) as evaluator:
		results = evaluator.perform_experiments(num_experiments)

	return results