# Brodderick Rodriguez
# Auburn University - CSSE
# 27 Aug. 2019

import strategy_learning_system as sls
import numpy as np
import sys
from sklearn.metrics import auc


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
	world_edge_threshold = sls.RealParameter('world-edge-threshold', 0, 25, default=0)
	max_world_edge_turn = sls.RealParameter('max-world-edge-turn', 0, 20, default=20)
	territory = sls.Feature('territory')
	territory.add_sub_feature(world_edge_threshold, sls.FeatureType.model, sls.Constraint.optional)
	territory.add_sub_feature(max_world_edge_turn, sls.FeatureType.model, sls.Constraint.optional)

	# environment wind
	wind_speed = sls.RealParameter('wind-speed', 0, 0.1, default=0)
	wind_heading = sls.IntegerParameter('wind-heading', 0, 360, default=0)
	wind = sls.Feature('wind')
	wind.add_sub_feature(wind_speed, sls.FeatureType.environmental, sls.Constraint.mandatory)
	wind.add_sub_feature(wind_heading, sls.FeatureType.environmental, sls.Constraint.mandatory)

	# environment plume characteristics
	plume_spread = sls.RealParameter('plume-spread-radius', 0, 1, default=0.25)
	plume_decay = sls.RealParameter('plume-decay-rate', 0, 1e-4, default=0)
	plume_decon_threshold = sls.RealParameter('plume-decontamination-threshold', 0.01, 1.0, default=0)
	plume_characteristics = sls.Feature('plume-characteristics')
	plume_characteristics.add_sub_feature(plume_spread, sls.FeatureType.environmental, sls.Constraint.mandatory)
	plume_characteristics.add_sub_feature(plume_decay, sls.FeatureType.environmental, sls.Constraint.optional)
	plume_characteristics.add_sub_feature(plume_decon_threshold, sls.FeatureType.environmental, sls.Constraint.optional)

	# environment
	environment = sls.Feature('environment')
	num_plumes = sls.IntegerParameter('number-plumes', 1, 5, default=1)
	environment.add_sub_feature(num_plumes, sls.FeatureType.environmental, sls.Constraint.mandatory)
	environment.add_sub_feature(territory, sls.FeatureType.model, sls.Constraint.optional)
	environment.add_sub_feature(wind, sls.FeatureType.environmental, sls.Constraint.optional)
	environment.add_sub_feature(plume_characteristics, sls.FeatureType.environmental, sls.Constraint.mandatory)

	# swarm UAV Capacity sensor reading
	coverage_per = sls.TimeSeriesOutcome('coverage-percentage')
	coverage_std = sls.TimeSeriesOutcome('coverage-std')
	coverage_mean = sls.TimeSeriesOutcome('coverage-mean')
	coverage = sls.Feature('coverage')
	coverage.add_sub_feature(coverage_per, sls.FeatureType.outcome, sls.Constraint.optional)
	coverage.add_sub_feature(coverage_std, sls.FeatureType.outcome, sls.Constraint.optional)
	coverage.add_sub_feature(coverage_mean, sls.FeatureType.outcome, sls.Constraint.optional)

	coverage_data_decay = sls.IntegerParameter('coverage-data-decay', 1, 60, default=60)
	sensor_reading = sls.Feature('sensor-reading')
	sensor_reading.add_sub_feature(coverage_data_decay, sls.FeatureType.model, sls.Constraint.optional)
	sensor_reading.add_sub_feature(coverage, sls.FeatureType.outcome, sls.Constraint.optional)

	# swarm uav capacity
	uav_vision = sls.RealParameter('UAV-vision', 0, 195, default=48)
	uav_decon_strength = sls.RealParameter('UAV-decontamination-strength', 0, 0.01, default=0)
	uav_capacity = sls.Feature('uav-capacity')
	uav_capacity.add_sub_feature(sensor_reading, sls.FeatureType.environmental, sls.Constraint.mandatory)
	uav_capacity.add_sub_feature(uav_vision, sls.FeatureType.environmental, sls.Constraint.optional)
	uav_capacity.add_sub_feature(uav_decon_strength, sls.FeatureType.environmental, sls.Constraint.optional)

	# swam search behavior
	search_lookup = {'flock': '\"flock-search\"', 'random': '\"random-search\"', 'symmetric': '\"symmetric-search\"'}

	# swarm search behavior flock policy
	min_sep = sls.RealParameter('minimum-separation', 0, 5, default=0.75)
	max_align = sls.RealParameter('max-align-turn', 0, 20, default=0.0)
	max_cohere = sls.RealParameter('max-cohere-turn', 0, 10, default=1.9)
	max_separate = sls.RealParameter('max-separate-turn', 0, 20, default=4.75)
	flock_policy = sls.Feature(search_lookup['flock'])
	flock_policy.add_sub_feature(min_sep, sls.FeatureType.model, sls.Constraint.mandatory)
	flock_policy.add_sub_feature(max_align, sls.FeatureType.model, sls.Constraint.mandatory)
	flock_policy.add_sub_feature(max_cohere, sls.FeatureType.model, sls.Constraint.mandatory)
	flock_policy.add_sub_feature(max_separate, sls.FeatureType.model, sls.Constraint.mandatory)

	# swarm search behavior random policy
	rand_max_heading_time = sls.IntegerParameter('random-search-max-heading-time', 0, 100, default=26)
	rand_max_turn = sls.RealParameter('random-search-max-turn', 0, 5, default=1.45)
	random_policy = sls.Feature(search_lookup['random'])
	random_policy.add_sub_feature(rand_max_heading_time, sls.FeatureType.model, sls.Constraint.mandatory)
	random_policy.add_sub_feature(rand_max_turn, sls.FeatureType.model, sls.Constraint.mandatory)

	# swarm search behavior symmetric policy
	sym_max_turn = sls.RealParameter('symmetric-search-max-turn', 0, 20, default=4.0)
	sym_region_threshold = sls.RealParameter('symmetric-search-region-threshold', -10, 25, default=-0.5)
	sym_min_region_time = sls.IntegerParameter('symmetric-search-min-region-time', 1, int(1e3), default=50)
	sym_max_region_time = sls.IntegerParameter('symmetric-search-max-region-time', int(1e2), int(5e3), default=1800)
	symmetric_policy = sls.Feature(search_lookup['symmetric'])
	symmetric_policy.add_sub_feature(sym_max_turn, sls.FeatureType.model, sls.Constraint.mandatory)
	symmetric_policy.add_sub_feature(sym_region_threshold, sls.FeatureType.model, sls.Constraint.mandatory)
	symmetric_policy.add_sub_feature(sym_min_region_time, sls.FeatureType.model, sls.Constraint.mandatory)
	symmetric_policy.add_sub_feature(sym_max_region_time, sls.FeatureType.model, sls.Constraint.mandatory)

	#swarm search behavior
	search_behavior = sls.CategoricalParameter('global-search-policy', categories=list(search_lookup.values()), default=search_lookup['flock'])
	search_behavior.add_sub_feature(flock_policy, sls.FeatureType.model, sls.Constraint.xor, search_lookup['flock'])
	search_behavior.add_sub_feature(random_policy, sls.FeatureType.model, sls.Constraint.xor, search_lookup['random'])
	search_behavior.add_sub_feature(symmetric_policy, sls.FeatureType.model, sls.Constraint.xor, search_lookup['symmetric'])

	# swarm
	pop = sls.IntegerParameter('population', 2, 100, default=12)
	swarm = sls.Feature('swarm')
	swarm.add_sub_feature(uav_capacity, sls.FeatureType.environmental, sls.Constraint.mandatory)
	swarm.add_sub_feature(pop, sls.FeatureType.model, sls.Constraint.mandatory)
	swarm.add_sub_feature(search_behavior, sls.FeatureType.model, sls.Constraint.mandatory)

	# feature model
	feature_model = sls.FeatureModel()
	feature_model.add_sub_feature(environment, sls.FeatureType.environmental, sls.Constraint.mandatory)
	feature_model.add_sub_feature(swarm, sls.FeatureType.model, sls.Constraint.mandatory)
	return feature_model


def create_plume_mediator():
	mediator = sls.ModelMediator(name=MEDIATOR_NAME)
	mediator.model = (MODEL_DIR, MODEL_FILE_NAME)
	mediator.netlogo = (NETLOGO_HOME, '6.0')
	mediator.feature_model = define_feature_model()
	mediator.save_location = SAVE_LOC
	mediator.save()
	return mediator



# TODO: delete before publishing
# this has to be here because the pickled mediator is looking for it
# by name. It does not impact performance. Only causes a runtime error if removed.
def reward_function_2(outcome_keys, outcomes):
	pass


def area_under_curve(outcome_keys, outcomes):
	rewards = np.zeros((outcomes.shape[0]))
	x = np.arange(0, outcomes.shape[2])

	for i, experiment_outcomes in enumerate(outcomes):
		d = {key: exp_out for key, exp_out in zip(outcome_keys, experiment_outcomes)}
		cov_per = d['coverage-percentage']
		auc_i = auc(x, cov_per)
		rewards[i] = auc_i

	r_min = np.min(rewards)
	r_max = np.max(rewards)
	rewards = (rewards - r_min) / (r_max - r_min)
	return rewards


# plume_mediator = create_plume_mediator()
plume_mediator = sls.ModelMediator.load('{}/{}'.format(SAVE_LOC, MEDIATOR_NAME))



# VALIDATION EXPERIMENT 1
# zeta = 0.5
def create_validation_1(exp_name):
	resolution = []
	resolution.append(plume_mediator.features['population'])
	resolution.append(plume_mediator.features['number-plumes'])
	resolution.append(plume_mediator.features['coverage-percentage'])

	cxt = sls.Context(name=exp_name)
	cxt.reward_function = area_under_curve
	cxt.resolution_model = resolution

	cxt.bins = np.linspace(0, 1, 5)
	cxt.num_experiments = 30
	cxt.num_replications = 30
	cxt.max_run_length = 1000
	cxt.num_processes = 11
	return cxt


def run_validation_1():
	print('experiment: validation 1')
	exp_name = 'simple_valid_4'
	# cxt = create_validation_1(exp_name)
	# plume_mediator.evaluate_context(cxt)
	# plume_mediator.save()

	cxt = plume_mediator[exp_name]
	# plume_mediator.learn(cxt, algorithm='ann_hac')
	# plume_mediator.save()

	# plume_mediator.explain(cxt)

	print(plume_mediator.features)


# VALIDATION EXPERIMENT 2
# zeta = 0.25
def create_validation_2(exp_name):
	resolution = []
	resolution.append(plume_mediator.features['population'])
	resolution.append(plume_mediator.features['coverage-data-decay'])
	resolution.append(plume_mediator.features['wind-speed'])
	resolution.append(plume_mediator.features['number-plumes'])
	resolution.append(plume_mediator.features['coverage-percentage'])

	cxt = sls.Context(name=exp_name)
	cxt.reward_function = area_under_curve
	cxt.resolution_model = resolution

	cxt.bins = np.linspace(0.0, 1.0, 3)
	cxt.num_experiments = 62
	cxt.num_replications = 30
	cxt.max_run_length = 1000
	cxt.num_processes = 11
	return cxt


def run_validation_2():
	print('experiment: validation 2')
	exp_name = 'context2'

	# cxt = create_validation_2(exp_name)
	# plume_mediator.evaluate_context(cxt)
	# plume_mediator.save()

	cxt = plume_mediator[exp_name]
	plume_mediator.learn(cxt, algorithm='ann_hac')
	# plume_mediator.save()

	plume_mediator.explain(cxt)


# EXPLORATORY EXPERIMENT 1
# zeta = 0.25
def create_exploratory_1(exp_name):
	resolution = []
	resolution.append(plume_mediator.features['\"symmetric-search\"'])
	resolution.append(plume_mediator.features['number-plumes'])
	resolution.append(plume_mediator.features['UAV-vision'])
	resolution.append(plume_mediator.features['coverage-percentage'])

	cxt = sls.Context(name=exp_name)
	cxt.reward_function = area_under_curve
	cxt.resolution_model = resolution

	cxt.bins = np.linspace(0.0, 1.0, 3)
	cxt.num_experiments = 150
	cxt.num_replications = 30
	cxt.max_run_length = 1000
	cxt.num_processes = 12
	return cxt


def run_exploratory_1():
	print('experiment: exploratory 1')
	exp_name = 'exploratory_exp_1_new'
	# cxt = create_exploratory_1(exp_name)
	# plume_mediator.evaluate_context(cxt)
	# plume_mediator.save()

	cxt = plume_mediator[exp_name]
	# plume_mediator.learn(cxt)
	# plume_mediator.save()

	plume_mediator.explain(cxt)


# EXPLORATORY EXPERIMENT 2
# zeta = 0.25
def create_exploratory_2(exp_name):
	resolution = []
	resolution.append(plume_mediator.features['\"flock-search\"'])
	resolution.append(plume_mediator.features['number-plumes'])
	resolution.append(plume_mediator.features['UAV-vision'])
	resolution.append(plume_mediator.features['coverage-percentage'])

	cxt = sls.Context(name=exp_name)
	cxt.reward_function = area_under_curve
	cxt.resolution_model = resolution

	cxt.bins = np.linspace(0.0, 1.0, 3)
	cxt.num_experiments = 200
	cxt.num_replications = 30
	cxt.max_run_length = 1000
	cxt.num_processes = 12
	return cxt


def run_exploratory_2():
	print('experiment: exploratory 2')
	exp_name = 'exploratory_exp_4'
	# cxt = create_exploratory_2(exp_name)
	# plume_mediator.evaluate_context(cxt)
	# plume_mediator.save()

	cxt = plume_mediator[exp_name]
	# plume_mediator.learn(cxt)
	# plume_mediator.save()

	plume_mediator.explain(cxt)


# EXPLORATORY EXPERIMENT Search Policy
# zeta = 0.25
def create_exploratory_search_policy(exp_name):
	resolution = []
	resolution.append(plume_mediator.features.get_item('global-search-policy', include_children=False))
	resolution.append(plume_mediator.features['population'])
	resolution.append(plume_mediator.features['number-plumes'])
	resolution.append(plume_mediator.features['wind-speed'])
	resolution.append(plume_mediator.features['coverage-percentage'])

	cxt = sls.Context(name=exp_name)
	cxt.reward_function = area_under_curve
	cxt.resolution_model = resolution

	cxt.bins = np.linspace(0.0, 1.0, 3)
	cxt.num_experiments = 50
	cxt.num_replications = 10
	cxt.max_run_length = 1000
	cxt.num_processes = 12
	return cxt


def run_exploratory_search_policy():
	print('experiment: exploratory search policy')
	exp_name = 'exploratory_exp_search_policy_5'
	# cxt = create_exploratory_search_policy(exp_name)
	# plume_mediator.evaluate_context(cxt)
	# plume_mediator.save()

	cxt = plume_mediator[exp_name]
	plume_mediator.learn(cxt, algorithm='ann_hac')
	plume_mediator.save()

	plume_mediator.explain(cxt)


# TODO: experiment not completed due to scope of research
def create_generalization_1(exp_name):
	resolution = []
	resolution.append(plume_mediator.features['population'])
	resolution.append(plume_mediator.features['coverage-data-decay'])
	resolution.append(plume_mediator.features['wind-speed'])
	resolution.append(plume_mediator.features['number-plumes'])
	resolution.append(plume_mediator.features['coverage-percentage'])

	cxt = sls.Context(name=exp_name)
	cxt.reward_function = area_under_curve
	cxt.resolution_model = resolution

	cxt.bins = np.linspace(0.0, 1.0, 3)
	cxt.num_experiments = 162
	cxt.num_replications = 30
	cxt.max_run_length = 1000
	cxt.num_processes = 11
	return cxt


# TODO: experiment not completed due to scope of research
def run_generalization_1():
	print('experiment: generalization 1')
	exp_name = 'exploratory_exp_3_new'
	# cxt = create_exploratory_3(exp_name)
	# plume_mediator.evaluate_context(cxt)
	# plume_mediator.save()

	cxt = plume_mediator[exp_name]
	# plume_mediator.learn(cxt)
	# plume_mediator.save()

	plume_mediator.explain(cxt)


if __name__ == '__main__':
	print('SLS: contaminant plume model')
	run_validation_1()
	# run_validation_2()
	# run_exploratory_1()
	# run_exploratory_2()
	# run_exploratory_search_policy()
	# run_generalization_1()
	pass
