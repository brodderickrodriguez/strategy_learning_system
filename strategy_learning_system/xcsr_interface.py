# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019

import xcsr
from .feature import IntegerParameter
from .rule import Rule
import numpy as np
import shutil


class GenericConfiguration(xcsr.Configuration):
	def __init__(self):
		xcsr.Configuration.__init__(self)

		self.alpha = 0.1
		self.epsilon_0 = 0.01
		self.v = 5

		self.gamma = np.random.uniform(0.9, 0.99)

		self.theta_mna = 2

		self.do_ga_subsumption = True

		# length of an episode
		self.steps_per_episode = 10 ** 4 * 2
		self.episodes_per_replication = 1
		self.is_multi_step = False

		self.beta = 0.9

		self.predicate_1 = 0.0
		self.p_1 = 0.1
		self.epsilon_1 = np.random.uniform(0, 10 ** -4)
		self.F_1 = 0.1

		# the maximum size of the population (in micro-classifiers)
		self.N = 50

		# the GA threshold. GA is applied in a set when the average time
		# since the last GA in the set is greater than theta_ga
		self.theta_ga = 50

		# the probability of applying crossover in the GA
		self.chi = 0.15

		# specifies the probability of mutating an allele in the offspring
		self.mu = np.random.uniform(0.001, 0.01)
		# self.mu = 0.25

		# subsumption threshold. experience of a classifier must be greater
		# than theta_sub in order to be able to subsume another classifier
		self.theta_sub = 20

		# probability of using '#' (Classifier.WILDCARD_ATTRIBUTE_VALUE)
		# in one attribute in the condition of a classifier when covering
		self.p_sharp = 0.5

		# probability during action selection of choosing the
		# action uniform randomly
		self.p_explr = 1.0


class GenericEnvironment(xcsr.Environment):
	def __init__(self, config, *args):
		xcsr.Environment.__init__(self, config)

		self._current_state_idx = 0
		self.states, self.actions, self.rhos = args[0]
		self.state_shape = (self.states.shape[1],)
		self.action_shape = (self.actions.shape[1],)

		self.possible_actions = self._get_possible_actions()

		self.max_value = self._find_max_value(self.states, self.actions)
		self.max_alpha_norm = np.linalg.norm([self.max_value for _ in range(self.state_shape[0])])


		self.states = self.states.to_numpy()
		self.actions = self.actions.to_numpy()
		self.rhos = self.rhos.to_numpy()

		self._set_state()

	@staticmethod
	def _find_max_value(states, actions):
		s = np.amax(np.array(states))
		a = np.amax(np.array(actions))
		return np.max([s, a])

	def _get_possible_actions(self):
		actions = []
		for action in self.actions.drop_duplicates().to_numpy():
			actions.append(tuple(action))
		return actions

	def get_state(self):
		return self._state

	def _set_state(self):
		self._current_state_idx = np.random.choice(len(self.states))
		self._state = np.array(self.states[self._current_state_idx])

	def step(self, action):
		self.end_of_program = True
		self.time_step += 1
		rho = self._determine_rho(action)
		self._set_state()
		return rho

	def _determine_rho(self, action):
		alpha_hat = np.array(action)

		alpha = np.array(self.actions[self._current_state_idx])

		step_alpha_norm = np.linalg.norm(alpha_hat - alpha)

		step_alpha_delta = step_alpha_norm / self.max_alpha_norm

		rho = self.rhos[self._current_state_idx]

		zeta = 0.5

		rho_hat = rho - (zeta * rho * step_alpha_delta)

		return rho_hat

	def termination_criteria_met(self):
		return self.time_step >= self._max_steps

	def print_world(self):
		print('state:\t\t{}\nexp_action:\t{}\n'.format(self._state, np.array(self.actions.loc[self._current_state_idx])))


def _parse_data(feature_model, resolution_model, data):
	environmental_names = [eu.name for eu in feature_model.environmental_uncertainties()]
	model_names = [mu.name for mu in feature_model.model_uncertainties()]
	resolution_names = [rmu.name for rmu in resolution_model]

	env_uncertainties = list(set(environmental_names) & set(resolution_names))
	mod_uncertainties = list(set(model_names) & set(resolution_names))

	states = data[env_uncertainties]
	actions = data[mod_uncertainties]
	rhos = data['rho']

	states_np = np.array(states)
	state_min = np.min(states_np)
	state_max = np.max(states_np)

	states = (states - state_min) / (state_max - state_min)

	return states, actions, rhos


def _run_xcsr(env, config, data, save_loc):
	driver = xcsr.XCSRDriver()
	driver.config_class = config
	driver.env_class = env
	driver.env_args = data
	driver.replications = 2
	driver.save_location = save_loc
	driver.experiment_name = 'learned_data'
	classifiers = driver.run()

	dir_name = '{}/{}'.format(driver.save_location, driver.experiment_name)
	fig = xcsr.util.plot_results(dir_name, title='', interval=50, generate_only=True)
	shutil.rmtree(dir_name)
	return classifiers, fig


def _rule_from_classifier(classifier, env_uncertainty, mod_uncertainty, bins):
	rule = Rule(classifier)
	rule.outcome = classifier.predicted_payoff
	rule.confidence = 1 - classifier.epsilon
	rule.experience = classifier.experience

	def _get_true_value(f_range, uncertainty):
		r = lambda p: p * (uncertainty.upper_bound - uncertainty.lower_bound) + uncertainty.lower_bound
		a, b = r(f_range[0]), r(f_range[1])

		if isinstance(uncertainty, IntegerParameter):
			a, b = int(a), int(b)

		return a, b

	def _get_bin_range(a):
		a = int(a)
		l = a - 1 if a > 0 else 0
		r = a if a != 0 else 1
		return bins[l], bins[r]

	for pred, eu in zip(classifier.predicate, env_uncertainty):
		p_true = _get_true_value(pred, eu)
		rule.environmental_uncertainties.append((eu, p_true, pred))

	for act, mu in zip(classifier.action, mod_uncertainty):
		act_range = _get_bin_range(act)
		act_true = _get_true_value(act_range, mu)
		rule.model_uncertainties.append((mu, act_true, act_range))

	return rule


def _make_rules(cxt, classifiers):
	env_uncertainty = cxt.environmental_uncertainties()
	mod_uncertainty = cxt.model_uncertainties()
	classifiers = [cl for rep in classifiers for cl in rep]
	rules = []

	for cl in classifiers:
		if cl.experience == 0:
			continue
		else:
			r = _rule_from_classifier(cl, env_uncertainty, mod_uncertainty, cxt.bins)
			rules.append(r)

	return rules


def learn(mediator, cxt):
	save_loc = '{}/{}'.format(mediator.save_location, mediator.name)
	feature_model = mediator.feature_model
	resolution_model = cxt.collapsed_resolution_model()
	exploratory_data = cxt.exploratory_data

	data = _parse_data(feature_model=feature_model,
					   resolution_model=resolution_model,
					   data=exploratory_data)

	config = GenericConfiguration
	env = GenericEnvironment
	classifiers, fig = _run_xcsr(env, config, data, save_loc)
	fig.savefig('{}/learn_plot.png'.format(save_loc))

	rules = _make_rules(cxt, classifiers)
	return rules