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
	# environment territory
	world_edge_threshold = sls.RealParameter('world-edge-threshold', 0, 25)
	max_world_edge_turn = sls.RealParameter('max-world-edge-turn', 0, 20)
	territory = sls.Feature(feature_name='territory')
	territory.add_sub_feature(world_edge_threshold, sls.FeatureType.model, sls.Constraint.optional)
	territory.add_sub_feature(max_world_edge_turn, sls.FeatureType.model, sls.Constraint.optional)

	# environment wind
	wind_speed = sls.RealParameter('wind-speed', 0, 0.1)
	wind_heading = sls.IntegerParameter('wind-heading', 0, 360)
	wind = sls.Feature(feature_name='wind')
	wind.add_sub_feature(wind_speed, sls.FeatureType.environmental, sls.Constraint.mandatory)
	wind.add_sub_feature(wind_heading, sls.FeatureType.environmental, sls.Constraint.mandatory)

	# environment plume characteristics
	plume_spread = sls.RealParameter('plume-spread-radius', 0, 1)
	plume_decay = sls.RealParameter('plume-decay-rate', 0, 1e-4)
	plume_decon_threshold = sls.RealParameter('plume-decontamination-threshold', 0.01, 1)
	plume_characteristics = sls.Feature(feature_name='plume characteristics')
	plume_characteristics.add_sub_feature(plume_spread, sls.FeatureType.environmental, sls.Constraint.mandatory)
	plume_characteristics.add_sub_feature(plume_decay, sls.FeatureType.environmental, sls.Constraint.optional)
	plume_characteristics.add_sub_feature(plume_decon_threshold, sls.FeatureType.environmental, sls.Constraint.optional)

	# environment
	environment = sls.Feature(feature_name='environment')
	num_plumes = sls.IntegerParameter('number-plumes', 0, 5)
	environment.add_sub_feature(num_plumes, sls.FeatureType.environmental, sls.Constraint.mandatory)
	environment.add_sub_feature(territory, sls.FeatureType.environmental, sls.Constraint.optional)
	environment.add_sub_feature(wind, sls.FeatureType.environmental, sls.Constraint.optional)
	environment.add_sub_feature(plume_characteristics, sls.FeatureType.environmental, sls.Constraint.mandatory)

	# swarm UAV Capacity sensor reading
	coverage_data_decay = sls.IntegerParameter('coverage-data-decay', 1, 60)
	coverage_per = sls.TimeSeriesOutcome('coverage-percentage')
	coverage_std = sls.TimeSeriesOutcome('coverage-std')
	coverage_mean = sls.TimeSeriesOutcome('coverage-mean')
	sensor_reading = sls.Feature(feature_name='sensor reading')
	sensor_reading.add_sub_feature(coverage_data_decay, sls.FeatureType.model, sls.Constraint.optional)
	sensor_reading.add_sub_feature(coverage_per, sls.FeatureType.outcome, sls.Constraint.optional)
	sensor_reading.add_sub_feature(coverage_std, sls.FeatureType.outcome, sls.Constraint.optional)
	sensor_reading.add_sub_feature(coverage_mean, sls.FeatureType.outcome, sls.Constraint.optional)

	# swarm uav capacity
	uav_vision = sls.RealParameter('UAV-vision', 0, 195)
	uav_decon_strength = sls.RealParameter('UAV-decontamination-strength', 0, 0.01)
	uav_capacity = sls.Feature(feature_name='uav capacity')
	uav_capacity.add_sub_feature(sensor_reading, sls.FeatureType.outcome, sls.Constraint.mandatory)
	uav_capacity.add_sub_feature(uav_vision, sls.FeatureType.environmental, sls.Constraint.optional)
	uav_capacity.add_sub_feature(uav_decon_strength, sls.FeatureType.environmental, sls.Constraint.optional)

	# swarm search behavior flock policy
	min_sep = sls.RealParameter('minimum-separation', 0, 5)
	max_align = sls.RealParameter('max-align-turn', 0, 20)
	max_cohere = sls.RealParameter('max-cohere-turn', 0, 10)
	max_separate = sls.RealParameter('max-separate-turn', 0, 20)
	flock_policy = sls.Feature(feature_name='flock search policy')
	flock_policy.add_sub_feature(min_sep, sls.FeatureType.model, sls.Constraint.mandatory)
	flock_policy.add_sub_feature(max_align, sls.FeatureType.model, sls.Constraint.mandatory)
	flock_policy.add_sub_feature(max_cohere, sls.FeatureType.model, sls.Constraint.mandatory)
	flock_policy.add_sub_feature(max_separate, sls.FeatureType.model, sls.Constraint.mandatory)

	# swarm search behavior random policy
	rand_max_heading_time = sls.IntegerParameter('random-search-max-heading-time', 0, 100)
	rand_max_turn = sls.RealParameter('random-search-max-turn', 0, 5)
	random_policy = sls.Feature(feature_name='random search policy')
	random_policy.add_sub_feature(rand_max_heading_time, sls.FeatureType.model, sls.Constraint.mandatory)
	random_policy.add_sub_feature(rand_max_turn, sls.FeatureType.model, sls.Constraint.mandatory)

	# swarm search behavior symmetric policy
	sym_max_turn = sls.RealParameter('symmetric-search-max-turn', 0, 20)
	sym_region_threshold = sls.RealParameter('symmetric-search-region-threshold', -10, 25)
	sym_min_region_time = sls.IntegerParameter('symmetric-search-min-region-time', 1, int(1e3))
	sym_max_region_time = sls.IntegerParameter('symmetric-search-max-region-time', int(1e2), int(5e3))
	symmetric_policy = sls.Feature(feature_name='symmetric search policy')
	symmetric_policy.add_sub_feature(sym_max_turn, sls.FeatureType.model, sls.Constraint.mandatory)
	symmetric_policy.add_sub_feature(sym_region_threshold, sls.FeatureType.model, sls.Constraint.mandatory)
	symmetric_policy.add_sub_feature(sym_min_region_time, sls.FeatureType.model, sls.Constraint.mandatory)
	symmetric_policy.add_sub_feature(sym_max_region_time, sls.FeatureType.model, sls.Constraint.mandatory)

	#swarm search behavior
	search_behavior = sls.Feature(netlogo_categorical='global-search-policy', feature_name='search behavior')
	search_behavior.add_sub_feature(flock_policy, sls.FeatureType.model, sls.Constraint.xor, '\"flock-search\"')
	search_behavior.add_sub_feature(random_policy, sls.FeatureType.model, sls.Constraint.xor, '\"random-search\"')
	search_behavior.add_sub_feature(symmetric_policy, sls.FeatureType.model, sls.Constraint.xor, '\"flock-search\"')

	# swarm
	pop = sls.IntegerParameter('population', 2, 100)
	swarm = sls.Feature(feature_name='swarm')
	swarm.add_sub_feature(uav_capacity, sls.FeatureType.environmental, sls.Constraint.mandatory)
	swarm.add_sub_feature(pop, sls.FeatureType.model, sls.Constraint.mandatory)
	swarm.add_sub_feature(search_behavior, sls.FeatureType.model, sls.Constraint.mandatory)

	# feature model
	feature_model = sls.FeatureModel()
	feature_model.add_sub_feature(environment, sls.FeatureType.environmental, sls.Constraint.mandatory)
	feature_model.add_sub_feature(swarm, sls.FeatureType.environmental, sls.Constraint.mandatory)
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
	# print(cxt1_resolution)

	cxt.bins = np.linspace(0.0, 1.0, 3)
	cxt.num_experiments = 5
	cxt.num_replications = 10
	cxt.max_run_length = 5000
	cxt.num_processes = 3
	return cxt



med = create()

print(med.feature_model)

env = med.feature_model['search behavior']

print(med.feature_model.subtree_str(env))

print(med.feature_model.collapse(env))

# cxt1 = create_context1(med)
# med.evaluate_context(cxt1)
# med.save()

#
# med = sls.ModelMediator.load('{}/{}'.format(SAVE_LOC, MEDIATOR_NAME))
# cxt1 = med['context1']
# print(cxt1.processed_exploratory_results)

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







