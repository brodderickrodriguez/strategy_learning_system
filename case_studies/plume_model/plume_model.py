# Brodderick Rodriguez
# Auburn University - CSSE
# 27 Aug. 2019

import strategy_learning_system as sls
import numpy as np
import sys


# for bcr 
if sys.platform == 'darwin':
	ROOT = '/Users/bcr/Dropbox/projects'
	NETLOGO_HOME = '/Applications/NetLogo-6.0.4/'
else:
	ROOT = '/home/bcr/Dropbox/projects'
	NETLOGO_HOME = '/home/bcr/apps/NetLogo 6.0.4/'

MODEL_DIR = ROOT + '/code/scala/Plume-Model/Scala-Plume-Model/nlogo-model'
MODEL_FILE_NAME = 'plume_extended.nlogo'
MEDIATOR_NAME = 'plume'
SAVE_LOC = ROOT + '/data/sls_data'


def define_feature_model():
	# environmental uncertainties
	num_plumes = sls.IntegerParameter('number-plumes', 0, 5)
	plume_spread = sls.RealParameter('plume-spread-radius', 0, 1)
	plume_decay = sls.RealParameter('plume-decay-rate', 0, 1e-4)
	plume_decon_threshold = sls.RealParameter('plume-decontamination-threshold', 0.01, 1)
	wind_speed = sls.RealParameter('wind-speed', 0, 0.1)
	wind_heading = sls.IntegerParameter('wind-heading', 0, 360)
	coverage_data_decay = sls.IntegerParameter('coverage-data-decay', 1, 60)

	# model uncertainties
	pop = sls.IntegerParameter('population', 2, 100)
	uav_vision = sls.RealParameter('UAV-vision', 0, 195)
	uav_decon_strength = sls.RealParameter('UAV-decontamination-strength', 0, 0.01)
	world_edge_threshold = sls.RealParameter('world-edge-threshold', 0, 25)
	max_world_edge_turn = sls.RealParameter('max-world-edge-turn', 0, 20)

	search_policy = sls.CategoricalParameter('global-search-policy',
	categories=('\"random-search\"', '\"flock-search\"', '\"symmetric-search\"'))

	# flocking policy
	min_sep = sls.RealParameter('minimum-separation', 0, 5)
	max_align = sls.RealParameter('max-align-turn', 0, 20)
	max_cohere = sls.RealParameter('max-cohere-turn', 0, 10)
	max_separate = sls.RealParameter('max-separate-turn', 0, 20)

	# random policy
	rand_max_heading_time = sls.IntegerParameter('random-search-max-heading-time', 0, 100)
	rand_max_turn = sls.RealParameter('random-search-max-turn', 0, 5)

	# symmetric policy
	sym_max_turn = sls.RealParameter('symmetric-search-max-turn', 0, 20)
	sym_region_threshold = sls.RealParameter('symmetric-search-region-threshold', -10, 25)
	sym_min_region_time = sls.IntegerParameter('symmetric-search-min-region-time', 1, int(1e3))
	sym_max_region_time = sls.IntegerParameter('symmetric-search-max-region-time', int(1e2), int(5e3))

	# outcomes
	coverage_per = sls.TimeSeriesOutcome('coverage-percentage')
	coverage_std = sls.TimeSeriesOutcome('coverage-std')
	coverage_mean = sls.TimeSeriesOutcome('coverage-mean')

	environmental_uncertainties = [num_plumes, plume_spread, plume_decay,
									plume_decon_threshold, wind_speed,
									wind_heading, coverage_data_decay]

	model_uncertainties = [search_policy, pop, uav_vision, uav_decon_strength,
							world_edge_threshold, max_world_edge_turn,
							min_sep, max_align, max_cohere,
							max_separate, rand_max_heading_time, rand_max_turn,
							sym_max_turn, sym_region_threshold,
							sym_min_region_time, sym_max_region_time]

	outcomes = [coverage_per, coverage_std, coverage_mean]

	feature_model = sls.FeatureModel()
	feature_model.environmental_uncertainties = environmental_uncertainties
	feature_model.model_uncertainties = model_uncertainties
	feature_model.outcomes = outcomes

	return feature_model


def create():
	_med = sls.ModelMediator(name=MEDIATOR_NAME)
	_med.model = (MODEL_DIR, MODEL_FILE_NAME)
	_med.netlogo = (NETLOGO_HOME, '6.0')
	_med.feature_model = define_feature_model()
	_med.save_location = SAVE_LOC
	_med.save()
	return _med


def reward_function_1(outcome_keys, outcomes):
	rewards = np.zeros((outcomes.shape[0]))

	for i, experiment_outcomes in enumerate(outcomes):
		d = {key: exp_out for key, exp_out in zip(outcome_keys, experiment_outcomes)}
		rho = d['coverage-percentage'][-1]
		rewards[i] = rho

	return rewards


def create_context1(mediator):
	cxt1_resolution = []
	cxt1_resolution.append(mediator.feature_model['coverage-percentage'])
	cxt1_resolution.append(mediator.feature_model['population'])
	cxt1_resolution.append(mediator.feature_model['global-search-policy'])
	# cxt1_resolution.append(mediator.feature_model['UAV-vision'])
	# cxt1_resolution.append(mediator.feature_model['wind-speed'])
	cxt1_resolution.append(mediator.feature_model['number-plumes'])

	cxt = sls.Context(name='context1')
	cxt.reward_function = reward_function_1
	cxt.resolution_model = cxt1_resolution
	# print(list(mediator.feature_model))
	print(cxt1_resolution)

	cxt.bins = np.linspace(0.0, 1.0, 3)
	cxt.num_experiments = 5
	cxt.num_replications = 10
	cxt.max_run_length = 5000
	cxt.num_processes = 3
	return cxt



med = create()
cxt1 = create_context1(med)
med.evaluate_context(cxt1)
med.save()

#
# med = sls.ModelMediator.load('{}/{}'.format(SAVE_LOC, MEDIATOR_NAME))
cxt1 = med['context1']
print(cxt1.processed_exploratory_results)

# print(med.feature_model['global-search-policy']._categories)

# states, actions, rhos = med.learn(cxt1)
# print(cxt1.resolution_model)

# print('rhos')
# print(rhos)

# with open('/Users/bcr/Dropbox/projects/data/sls_data/plume/learned_data/classifiers/replication0', 'rb') as f:
# 	classifiers = pickle.load(f)
# 	print(classifiers)
	#
	#
	# all = ''
	# header = 'predicate,action,pred,error,fit'
	#
	# all += header + '\n'
	#
	# for cl in classifiers:
	# 	s = '{},{},{},{},{}'.format(cl.predicate, cl.action, cl.predicted_payoff, cl.epsilon, cl.fitness)
	# 	all += s + '\n'
	#
	# print(all)

# def print_classifier(cl1):
# 	val = lambda p, x: abs((p.upper_bound * x) - p.lower_bound)
#
# 	print('classifier: ', cl1._id)
# 	print('\texpected reward: {}'.format(cl1.predicted_payoff))
# 	print('\terror: {}'.format(cl1.epsilon))
# 	print('\tenvironmental uncertainties:')
# 	# print(states.columns)
# 	for i in range(len(states.columns)):
# 		# print(i, states.columns[i])
# 		# print(cxt1[states.columns[i]])
#
# 		pram = cxt1[states.columns[i]]
# 		clp = cl1.predicate[i]
# 		lbr = clp[0] if clp[0] < 1 else clp[0] - 1
# 		lb = val(pram, lbr)
# 		ub = val(pram, clp[1])
#
# 		print('\t\t{}: {} - {}'.format(pram.name, lb, ub))
#
# 	print('\n\tmodel uncertainties:')
# 	for i in range(len(actions.columns)):
# 		pram = cxt1[actions.columns[i]]
# 		clp = cl1.action[i]
# 		action_val = val(pram, clp)
# 		# lbr = clp[0] if clp[0] < 1 else clp[1] - clp[0]
# 		# lb = val(pram, lbr)
# 		# ub = val(pram, clp[1])
# 		print('\t\t{}: {}'.format(pram.name, action_val))
#
#
# cl1 = classifiers[0]
#
# for cl1 in classifiers:
# 	print()
# 	print_classifier(cl1)







