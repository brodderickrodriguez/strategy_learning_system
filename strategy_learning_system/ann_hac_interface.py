# Brodderick Rodriguez
# Auburn University - CSSE
# 23 Feb. 2020

import numpy as np
import itertools
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import AgglomerativeClustering
from .rule import Rule
from .feature_model import IntegerParameter, CategoricalParameter



def _make_net_data(cxt):
	exploratory_data = cxt.processed_exploratory_results

	X_train = exploratory_data.drop('rho', axis=1).to_numpy()
	y_train = exploratory_data['rho'].to_numpy()

	num_features = len(exploratory_data.columns) - 1
	r = range(cxt.bins.shape[0])
	X_test = np.array(list(itertools.product(r, repeat=num_features)))

	return X_train, y_train, X_test


def _extrapolate(cxt):
	exploratory_data = cxt.processed_exploratory_results
	feature_names = exploratory_data.columns[:-1]
	X_train, y_train, X_test = _make_net_data(cxt)

	net_model = MLPRegressor(hidden_layer_sizes=(50, 500, 100, 50), max_iter=10000)
	net_model.fit(X_train, y_train)
	y_test_hat = net_model.predict(X_test)

	column_map = {feature_names[i]: X_test[:,i] for i in range(len(feature_names))}
	column_map['rho'] = y_test_hat
	extrapolated_results = pd.DataFrame(column_map)
	return extrapolated_results


def _get_cluster_labels(extrapolated_results):
	rhos = extrapolated_results['rho'].to_numpy().reshape(-1, 1)
	rho_std = np.std(rhos) * 2
	km = AgglomerativeClustering(distance_threshold=rho_std, n_clusters=None)
	cluster_labels = km.fit_predict(rhos)
	return cluster_labels


def make_single_rule(extrapolated_results, s, f, env_uncertainty, mod_uncertainty, bins):
	def _get_true_value(f_range, uncertainty):
		r = lambda p: p * (uncertainty.upper_bound - uncertainty.lower_bound) + uncertainty.lower_bound
		a, b = r(f_range[0]), r(f_range[1])

		if isinstance(uncertainty, IntegerParameter) or isinstance(uncertainty, CategoricalParameter):
			a, b = int(a), int(b)

		return a, b

	def _get_predicates(uncertainty):
		col_data = extrapolated_results.iloc[s:f][uncertainty.name]
		predicate_min = np.min(col_data) / bins.shape[0]
		predicate_max = np.max(col_data) / bins.shape[0] + (1/bins.shape[0])
		coded_predicate_ = predicate_min, predicate_max
		explainable_predicate_ = _get_true_value(coded_predicate_, uncertainty)
		return coded_predicate_, explainable_predicate_

	rule = Rule()
	rule.outcome = np.mean(extrapolated_results.iloc[s:f]['rho'])


	for eu in env_uncertainty:
		coded_predicate, explainable_predicate = _get_predicates(eu)
		rule.environmental_uncertainties.append((eu, explainable_predicate, coded_predicate))

	for mu in mod_uncertainty:
		coded_predicate, explainable_predicate = _get_predicates(mu)
		rule.model_uncertainties.append((mu, explainable_predicate, coded_predicate))

	return rule


def _make_rules(cxt, extrapolated_results, cluster_labels):
	env_uncertainty = cxt.environmental_uncertainties()
	mod_uncertainty = cxt.model_uncertainties()
	rules = []

	s, f, t = 0, 3, cluster_labels[0]
	for i in range(3, cluster_labels.shape[0]):
		if t != cluster_labels[i]:
			rule = make_single_rule(extrapolated_results, s, f, env_uncertainty, mod_uncertainty, cxt.bins)
			rules.append(rule)
			s = i
			t = cluster_labels[i]
		f += 1

	return rules


def learn(mediator, cxt):
	extrapolated_results = _extrapolate(cxt)

	cluster_labels = _get_cluster_labels(extrapolated_results)


	rules = _make_rules(cxt, extrapolated_results, cluster_labels)



	return rules
