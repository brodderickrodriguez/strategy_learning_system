# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import xcsr
from . import util
import numpy as np
import shutil


class GenericConfiguration(xcsr.Configuration):
	def __init__(self):
		xcsr.Configuration.__init__(self)

		# the maximum number of steps in each problem (replication)
		self.episodes_per_replication = 1

		# length of an episode
		self.steps_per_episode = 10 ** 2

		self.is_multi_step = False

		self.predicate_1 = 0.29

		self.predicate_delta = 0.1

		# the maximum size of the population (in micro-classifiers)
		self.N = 750


class GenericEnvironment(xcsr.Environment):
	def __init__(self, config, *args):
		print(config)
		xcsr.Environment.__init__(self, config)

		print(len(args))
		print(len(args[0]))

		self._current_state_idx = 0
		self.states, self.actions, self.rhos = args[0]
		self.state_shape = (self.states.shape[1],)
		self.action_shape = (self.actions.shape[1],)

		self.possible_actions = self._get_possible_actions()

		print(self.rhos)
		# exit()

		self._set_state()

	def _get_possible_actions(self):
		actions = []
		for action in self.actions.drop_duplicates().to_numpy():
			actions.append(tuple(action))
		return actions

	def get_state(self):
		return self._state

	def _set_state(self):
		self._current_state_idx = np.random.choice(len(self.states))
		print(self.states)
		self._state = np.array(self.states.iloc[0])

		print('state is ', self._state)

	def step(self, action):
		self.end_of_program = True
		self.time_step += 1
		rho = self._determine_rho(action)
		self._set_state()
		return rho

	def _determine_rho(self, action):
		actual_action = np.array(action)

		expected_action = np.array(self.actions.loc[self._current_state_idx])

		rmse = np.sum((actual_action - expected_action) ** 2)

		percentage = 1.0 if rmse == 0.0 else 1.0 / rmse

		print('exp: {} act: {} perc: {}'.format(expected_action, actual_action, percentage))

		return int(not False in expected_action == actual_action)

		return percentage * self.rhos[self._current_state_idx]

	def termination_criteria_met(self):
		return self.time_step >= self._max_steps

	def print_world(self):
		print('state:\t\t{}\nexp_action:\t{}\n'.format(self._state, np.array(self.actions.loc[self._current_state_idx])))


def _parse_data(feature_model, resolution_model, data):
	feature_env_uncertainties = [eu.name for eu in feature_model.environmental_uncertainties]
	feature_mod_uncertainties = [mu.name for mu in feature_model.model_uncertainties]
	resolution_model_names = [rmu.name for rmu in resolution_model]
	env_uncertainties = list(set(feature_env_uncertainties) & set(resolution_model_names))
	mod_uncertainties = list(set(feature_mod_uncertainties) & set(resolution_model_names))

	states = data[env_uncertainties]
	actions = data[mod_uncertainties]
	rhos = data['rho']

	return states, actions, rhos


def _run_xcsr(env, config, data, save_loc):
	driver = xcsr.XCSRDriver()
	driver.config_class = config
	driver.env_class = env
	driver.env_args = data
	driver.replications = 1
	driver.save_location = save_loc
	driver.experiment_name = 'learned_data'
	classifiers = driver.run()

	dir_name = '{}/{}'.format(driver.save_location, driver.experiment_name)
	xcsr.util.plot_results(dir_name, title='G', interval=10)
	# shutil.rmtree(dir_name)
	return classifiers



def learn(mediator, cxt):
	save_loc = '{}/{}'.format(mediator.save_location, mediator.name)
	print('mediator save location:', mediator.save_location)
	print(save_loc)

	data = _parse_data(mediator.feature_model, cxt.resolution_model, cxt.processed_exploratory_results)
	print(data, 'hi')
	return data

	# this looks like something for human play
	# config = GenericConfiguration
	# env = GenericEnvironment
	# _run_xcsr(env, config, data, save_loc)
	# env(config(), data).human_play()


if __name__ == '__main__':
	e = xcsr.Configuration()

