# Brodderick Rodriguez
# Auburn University - CSSE
# 27 Aug. 2019

import strategy_learning_system as sls
import numpy as np

ROOT = '/home/bcr/Dropbox/Projects'
MODEL_DIR = ROOT + '/CODE/NetLogo/prey_predator_nlogo'
MODEL_FILE_NAME = 'preypred.nlogo'
MEDIATOR_NAME = 'preypred'
SAVE_LOC = ROOT + '/data/sls_data'
NETLOGO_HOME = '/home/bcr/apps/NetLogo 6.0.4/'


def define_feature_model():
	initial_number_wolves = sls.IntegerParameter('initial-number-wolves', 1, 250)
	initial_number_sheep = sls.IntegerParameter('initial-number-sheep', 1, 250)
	grass_regrowth_time = sls.IntegerParameter('grass-regrowth-time', 1, 100)
	sheep_gain_food = sls.IntegerParameter('sheep-gain-from-food', 1, 50)
	wolf_gain_food = sls.IntegerParameter('wolf-gain-from-food', 1, 100)
	sheep_reproduce = sls.IntegerParameter('sheep-reproduce', 1, 20)
	wolf_reproduce = sls.IntegerParameter('wolf-reproduce', 1, 20)

	sheep_outcome = sls.TimeSeriesOutcome('sheep')
	wolves_outcome = sls.TimeSeriesOutcome('wolves')
	grass_outcome = sls.TimeSeriesOutcome('grass')
	ticks_outcome = sls.TimeSeriesOutcome('ticks')

	environmental_uncertainties = [grass_regrowth_time, sheep_gain_food, wolf_gain_food, sheep_reproduce, wolf_reproduce]
	model_uncertainties = [initial_number_wolves, initial_number_sheep]
	outcomes = [sheep_outcome, wolves_outcome, grass_outcome, ticks_outcome]

	feature_model = sls.FeatureModel()
	feature_model.environmental_uncertainties = environmental_uncertainties
	feature_model.model_uncertainties = model_uncertainties
	feature_model.outcomes = outcomes

	return feature_model


def create():
	med = sls.ModelMediator(name=MEDIATOR_NAME)
	med.model = (MODEL_DIR, MODEL_FILE_NAME)
	med.netlogo = (NETLOGO_HOME, '6.0')
	med.feature_model = define_feature_model()
	med.save_location = SAVE_LOC
	med.save()
	return med 	


def reward_function_1(outcomes, keys):
	MAX_TICK_ALLOWED = 100
	rewards = np.zeros((outcomes.shape[0]))

	for i, experiment_outcomes in enumerate(outcomes):
		d = {key: exp_out for key, exp_out in zip(keys, experiment_outcomes)}

		max_tick = np.max(d['ticks'])
		wolves_pop_std = np.std(d['wolves'])
		sheep_pop_std = np.std(d['sheep'])
		grass_pop_std = np.std(d['grass'])

		rho = wolves_pop_std + sheep_pop_std + grass_pop_std + (MAX_TICK_ALLOWED - max_tick)

		rewards[i] = (1.0 / rho) * 100

	return rewards


def create_context1(mediator):
	cxt1_resolution = mediator.feature_model.outcomes
	cxt1_resolution.append(mediator.feature_model['sheep-gain-from-food'])
	cxt1_resolution.append(mediator.feature_model['wolf-gain-from-food'])
	cxt1_resolution.append(mediator.feature_model['initial-number-sheep'])
	cxt1_resolution.append(mediator.feature_model['initial-number-wolves'])

	cxt = sls.Context(name='context1')
	cxt.reward_function = reward_function_1
	cxt.resolution_model = cxt1_resolution
	cxt.num_experiments = 10
	cxt.num_replications = 30
	cxt.max_run_length = 100
	cxt.num_processes = 3

	return cxt

def main():
	# mediator = create()
	mediator = sls.ModelMediator.load(root_dir_path=(SAVE_LOC + '/' + MEDIATOR_NAME))
	# mediator.save()

	cxt1 = create_context1(mediator)
	# mediator.evaluate_context(cxt1)

	# print(mediator)
	# print(cxt1)

	from ema_workbench import load_results
	path = '/home/bcr/Desktop/tmp.tar.gz'
	results = load_results(path)


	results = sls.util.process_ema_results(cxt1, results)

	rews = cxt1.reward_function(*results['outcomes'])
	print(rews)


main()

