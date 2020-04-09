# Brodderick Rodriguez
# Auburn University - CSSE
# 27 Aug. 2019

'''
NOTE to the user:
- The readme contains steps on how to configure SLS to run this experiment.
- Make sure the directory and file paths match your system. 
- Read the comments in __name__ == '__main__' to experiment with SLS correctly.
- Thanks!
'''

import strategy_learning_system as sls
import numpy as np
from sklearn.metrics import auc


def define_feature_model():
	# environment territory
	world_edge_threshold = sls.RealParameter('world-edge-threshold', 0, 25, default=0)
	max_world_edge_turn = sls.RealParameter('max-world-edge-turn', 0, 20, default=20)
	territory = sls.Feature('territory')
	territory.add_sub_feature(world_edge_threshold, sls.FeatureType.model, sls.FeatureConstraint.optional)
	territory.add_sub_feature(max_world_edge_turn, sls.FeatureType.model, sls.FeatureConstraint.optional)

	# environment wind
	wind_speed = sls.RealParameter('wind-speed', 0, 0.1, default=0)
	wind_heading = sls.IntegerParameter('wind-heading', 0, 360, default=0)
	wind = sls.Feature('wind')
	wind.add_sub_feature(wind_speed, sls.FeatureType.environmental, sls.FeatureConstraint.mandatory)
	wind.add_sub_feature(wind_heading, sls.FeatureType.environmental, sls.FeatureConstraint.mandatory)

	# environment plume characteristics
	plume_spread = sls.RealParameter('plume-spread-radius', 0, 1, default=0.25)
	plume_decay = sls.RealParameter('plume-decay-rate', 0, 1e-4, default=0)
	plume_decon_threshold = sls.RealParameter('plume-decontamination-threshold', 0.01, 1.0, default=0)
	plume_characteristics = sls.Feature('plume-characteristics')
	plume_characteristics.add_sub_feature(plume_spread, sls.FeatureType.environmental, sls.FeatureConstraint.mandatory)
	plume_characteristics.add_sub_feature(plume_decay, sls.FeatureType.environmental, sls.FeatureConstraint.optional)
	plume_characteristics.add_sub_feature(plume_decon_threshold, sls.FeatureType.environmental, sls.FeatureConstraint.optional)

	# environment
	environment = sls.Feature('environment')
	num_plumes = sls.IntegerParameter('number-plumes', 1, 5, default=1)
	environment.add_sub_feature(num_plumes, sls.FeatureType.environmental, sls.FeatureConstraint.mandatory)
	environment.add_sub_feature(territory, sls.FeatureType.model, sls.FeatureConstraint.optional)
	environment.add_sub_feature(wind, sls.FeatureType.environmental, sls.FeatureConstraint.optional)
	environment.add_sub_feature(plume_characteristics, sls.FeatureType.environmental, sls.FeatureConstraint.mandatory)

	# swarm UAV Capacity sensor reading
	coverage_per = sls.TimeSeriesOutcome('coverage-percentage')
	coverage_std = sls.TimeSeriesOutcome('coverage-std')
	coverage_mean = sls.TimeSeriesOutcome('coverage-mean')
	coverage = sls.Feature('coverage')
	coverage.add_sub_feature(coverage_per, sls.FeatureType.outcome, sls.FeatureConstraint.optional)
	coverage.add_sub_feature(coverage_std, sls.FeatureType.outcome, sls.FeatureConstraint.optional)
	coverage.add_sub_feature(coverage_mean, sls.FeatureType.outcome, sls.FeatureConstraint.optional)

	coverage_data_decay = sls.IntegerParameter('coverage-data-decay', 1, 60, default=60)
	sensor_reading = sls.Feature('sensor-reading')
	sensor_reading.add_sub_feature(coverage_data_decay, sls.FeatureType.model, sls.FeatureConstraint.optional)
	sensor_reading.add_sub_feature(coverage, sls.FeatureType.outcome, sls.FeatureConstraint.optional)

	# swarm uav capacity
	uav_vision = sls.RealParameter('UAV-vision', 0, 195, default=48)
	uav_decon_strength = sls.RealParameter('UAV-decontamination-strength', 0, 0.01, default=0)
	uav_capacity = sls.Feature('uav-capacity')
	uav_capacity.add_sub_feature(sensor_reading, sls.FeatureType.environmental, sls.FeatureConstraint.mandatory)
	uav_capacity.add_sub_feature(uav_vision, sls.FeatureType.environmental, sls.FeatureConstraint.optional)
	uav_capacity.add_sub_feature(uav_decon_strength, sls.FeatureType.environmental, sls.FeatureConstraint.optional)

	# swam search behavior
	search_lookup = {'flock': '\"flock-search\"', 'random': '\"random-search\"', 'symmetric': '\"symmetric-search\"'}

	# swarm search behavior flock policy
	min_sep = sls.RealParameter('minimum-separation', 0, 5, default=0.75)
	max_align = sls.RealParameter('max-align-turn', 0, 20, default=0.0)
	max_cohere = sls.RealParameter('max-cohere-turn', 0, 10, default=1.9)
	max_separate = sls.RealParameter('max-separate-turn', 0, 20, default=4.75)
	flock_policy = sls.Feature(search_lookup['flock'])
	flock_policy.add_sub_feature(min_sep, sls.FeatureType.model, sls.FeatureConstraint.mandatory)
	flock_policy.add_sub_feature(max_align, sls.FeatureType.model, sls.FeatureConstraint.mandatory)
	flock_policy.add_sub_feature(max_cohere, sls.FeatureType.model, sls.FeatureConstraint.mandatory)
	flock_policy.add_sub_feature(max_separate, sls.FeatureType.model, sls.FeatureConstraint.mandatory)

	# swarm search behavior random policy
	rand_max_heading_time = sls.IntegerParameter('random-search-max-heading-time', 0, 100, default=26)
	rand_max_turn = sls.RealParameter('random-search-max-turn', 0, 5, default=1.45)
	random_policy = sls.Feature(search_lookup['random'])
	random_policy.add_sub_feature(rand_max_heading_time, sls.FeatureType.model, sls.FeatureConstraint.mandatory)
	random_policy.add_sub_feature(rand_max_turn, sls.FeatureType.model, sls.FeatureConstraint.mandatory)

	# swarm search behavior symmetric policy
	sym_max_turn = sls.RealParameter('symmetric-search-max-turn', 0, 20, default=4.0)
	sym_region_threshold = sls.RealParameter('symmetric-search-region-threshold', -10, 25, default=-0.5)
	sym_min_region_time = sls.IntegerParameter('symmetric-search-min-region-time', 1, int(1e3), default=50)
	sym_max_region_time = sls.IntegerParameter('symmetric-search-max-region-time', int(1e2), int(5e3), default=1800)
	symmetric_policy = sls.Feature(search_lookup['symmetric'])
	symmetric_policy.add_sub_feature(sym_max_turn, sls.FeatureType.model, sls.FeatureConstraint.mandatory)
	symmetric_policy.add_sub_feature(sym_region_threshold, sls.FeatureType.model, sls.FeatureConstraint.mandatory)
	symmetric_policy.add_sub_feature(sym_min_region_time, sls.FeatureType.model, sls.FeatureConstraint.mandatory)
	symmetric_policy.add_sub_feature(sym_max_region_time, sls.FeatureType.model, sls.FeatureConstraint.mandatory)

	#swarm search behavior
	search_behavior = sls.CategoricalParameter('global-search-policy', categories=list(search_lookup.values()), default=search_lookup['flock'])
	search_behavior.add_sub_feature(flock_policy, sls.FeatureType.model, sls.FeatureConstraint.xor, search_lookup['flock'])
	search_behavior.add_sub_feature(random_policy, sls.FeatureType.model, sls.FeatureConstraint.xor, search_lookup['random'])
	search_behavior.add_sub_feature(symmetric_policy, sls.FeatureType.model, sls.FeatureConstraint.xor, search_lookup['symmetric'])

	# swarm
	pop = sls.IntegerParameter('population', 2, 100, default=12)
	swarm = sls.Feature('swarm')
	swarm.add_sub_feature(uav_capacity, sls.FeatureType.environmental, sls.FeatureConstraint.mandatory)
	swarm.add_sub_feature(pop, sls.FeatureType.model, sls.FeatureConstraint.mandatory)
	swarm.add_sub_feature(search_behavior, sls.FeatureType.model, sls.FeatureConstraint.mandatory)

	# feature model
	feature_model = sls.FeatureModel()
	feature_model.add_sub_feature(environment, sls.FeatureType.environmental, sls.FeatureConstraint.mandatory)
	feature_model.add_sub_feature(swarm, sls.FeatureType.model, sls.FeatureConstraint.mandatory)
	return feature_model


def create_plume_mediator(model_path, model_name, 
							netlogo_path, netlogo_v, 
							mediator_path, mediator_name):
	mediator = sls.ModelMediator(name=mediator_name)
	mediator.model = model_path, model_name
	mediator.netlogo = netlogo_path, netlogo_v
	mediator.feature_model = define_feature_model()
	mediator.save_location = mediator_path
	mediator.save()
	return mediator


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


def create_validation_1(plume_mediator):
	resolution = []
	resolution.append(plume_mediator.features['population'])
	resolution.append(plume_mediator.features['number-plumes'])
	resolution.append(plume_mediator.features['coverage-percentage'])

	cxt = sls.Context(name='validation_1')
	cxt.reward_function = area_under_curve
	cxt.resolution_model = resolution

	cxt.bins = np.linspace(0, 1, 5)
	cxt.num_experiments = 5
	cxt.num_replications = 1
	cxt.max_run_length = 10
	cxt.num_processes = 11
	return cxt


if __name__ == '__main__':
	# the path to the .nlogo model
	model_path = './contaminant_plume_model/Scala-Plume-Model/nlogo-model/'
	
	# the name of the .nlogo model
	model_name = 'plume_extended.nlogo'

	# the path to NetLogo. Note: make sure your NetLogo version is reflected here
	netlogo_path = '/Applications/NetLogo-6.0.4/'

	# the version of netlogo as a string. Specify only the major release version
	# I.e, '6.0', '7.0', etc.
	netlogo_version = '6.0'

	# the path to the mediator object that SLS will create and manage
	mediator_save_path = './experiment_data/'

	# the name of the mediator
	mediator_name = 'plume'
	
	# create the mediator object for the contaminant plume model
	# This line needs to be executed only once. It should be commented out after.
	# If not, it will overwrite any existing experiment data.
	plume_mediator = create_plume_mediator(model_path, model_name, 
											netlogo_path, netlogo_version, 
											mediator_save_path, mediator_name)

	# loads a previously created mediator
	# plume_mediator = sls.ModelMediator.load('{}/{}'.format(mediator_save_path, mediator_name))

	# create a context for a single experiment.
	# If you are performing multiple experiments, each experiment 
	# requires its own context.
	#
	# This is the process of experimenting with a single hypothesis about
	# the model.
	#
	# This line needs to be executed only once and should be commented
	# out afterwards. Otherwise, it will overwrite existing data. 
	cxt = create_validation_1(plume_mediator)

	# If you are experimenting with a context over multiple python executions,
	# you can load a previously created experiment using this line.
	# Note: a context needs to (at least) be explored before it will be saved.
	# cxt = plume_mediator['validation_1']

	# Execute the modeling via explroation and EMA 
	# This is the phase where we explicity evaluate many model instances 
	# and collect their performance using AUC
	plume_mediator.explore(cxt)

	# After exploration, we learn from the exploratory data. 
	# This is the processes of learning the patterns, and generating 
	# rules that explain the larger ensemble.
	# For this task, there are two algorithms: XCSR and MLP+HAC
	# plume_mediator.learn(cxt, algorithm='xcsr')
	plume_mediator.learn(cxt, algorithm='mlp_hac')

	# Next we can explain the context given our experiment hypothesis.
	# This will generate two heats maps: one for the explored data
	# and another for the learned rules.
	plume_mediator.explain(cxt)

	# Lastly, we save the mediator and its contexts.
	plume_mediator.save()

	print(plume_mediator, cxt)
	print('exploratory data:\n', cxt.exploratory_data)
	print('rules:\n', cxt.learned_data[:3])
