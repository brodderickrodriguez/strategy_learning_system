# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import xcsr
import numpy as np


class GenericConfiguration(xcsr.Configuration):
	def __init__(self):
		xcsr.Configuration.__init__(self)

		# the maximum number of steps in each problem (replication)
		self.episodes_per_replication = 1

		# length of an episode
		self.steps_per_episode = 10 

		self.is_multi_step = False

		self.predicate_1 = 0.29

		self.predicate_delta = 0.1

		# the maximum size of the population (in micro-classifiers)
		self.N = 750


class GenericEnvironment(xcsr.Environment):
	def __init__(self, config):
		xcsr.Environment.__init__(self, config, data)

		self._current_state_idx = 0
		self.states, self.actions, self.rhos = data
		self.state_shape = self.states.shape
		self.action_shape = self.actions.shape

		self._set_state()

	def get_state(self):
		return self._state

	def _set_state(self):
		self._current_state_idx = np.random.choice(len(self.states))
		self._state = np.array(self.states.loc[self._current_state_idx])

	def step(self, action):
		self.end_of_program = True
		self.time_step += 1
		rho = self._determine_rho(action)
		self._set_state()
		return rho

	def _determine_rho(self, action):
		actual_action = np.array(action)

		expected_action = np.array(self.actions.loc[self._current_state_idx])

		rmse = np.sqrt(np.sum((actual_action - expected_action) ** 2) / len(actual_action))

		percentage = 1.0 if rmse == 0 else 1.0 / rmse

		return percentage * self.rhos[self._current_state_idx]

	def termination_criteria_met(self):
		return self.time_step >= self._max_steps

	def print_world(self):
		print('state:\t\t{}\nexp_action:\t{}\n'.format(self._state, np.array(self.actions.loc[self._current_state_idx])))

	def human_play(self):
		while not self.termination_criteria_met():
			self.print_world()

			try:
				action = input('input action: ')
			except ValueError:
				print('invalid action')
				continue

			action = self.actions.loc[self._current_state_idx]

			rho = self.step(action)

			print('reward:\t', rho)
			print('eop?:\t', self.end_of_program)
			print()


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

def _run_xcsr(env, config, data):
	driver = xcsr.XCSRDriver()
	driver.config_class = config
	driver.env_class = env
	driver.env_args = data
	driver.replications = 5
	driver.save_location = '/Users/bcr/Desktop'
	driver.experiment_name = 'TMP2'
	driver.run()

	dir_name = '{}/{}'.format(driver.save_location, driver.experiment_name)
	xcsr.util.plot_results(dir_name, title='G', interval=50)
	shutil.rmtree(dir_name)


def learn(mediator, cxt):
	data = _parse_data(mediator.feature_model, cxt.resolution_model, cxt.processed_exploratory_results)

	config = GenericConfiguration

	env = GenericEnvironment

	print(cxt.processed_exploratory_results)

	_run_xcsr(env, config, data)

	env(config()).human_play()


if __name__ == '__main__':
	e = xcsr.Configuration()

