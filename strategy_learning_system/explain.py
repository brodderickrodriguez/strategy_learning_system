# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Dec. 2019


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import copy


def plot_explored(context):
	sns.set()
	# axis0: y - environmental
	# axis1: x - model

	y_delim, x_delim = ' ', '\n'

	environmental = [f.name for f in context.environmental_uncertainties()]
	model = [f.name for f in context.model_uncertainties()]

	data = context.processed_exploratory_results
	r = range(context.bins + 1) if isinstance(context.bins, int) else range(context.bins.shape[0])
	x_values = list(itertools.product(r, repeat=len(model)))
	y_values = list(itertools.product(r, repeat=len(environmental)))

	x_values = [x_delim.join(str(xi) for xi in x) for x in x_values]
	y_values = [y_delim.join(str(yi) for yi in y) for y in y_values]

	x_dict = {x: [] for x in x_values}
	structured_data = {y: copy.deepcopy(x_dict) for y in y_values[::-1]}

	def uncertainties_to_string(uncertainties, delim):
		n = [int(n) for n in uncertainties]
		s = delim.join(str(ni) for ni in n)
		return s

	for i in range(data.shape[0]):
		row = data.iloc[i]

		environmental_i = list(row[environmental])
		model_i = list(row[model])
		outcome = row['rho']

		data_i_idx = uncertainties_to_string(environmental_i, y_delim)
		data_j_idx = uncertainties_to_string(model_i, x_delim)
		structured_data[data_i_idx][data_j_idx].append(outcome)

	for i in structured_data:
		for j in structured_data[i]:
			structured_data[i][j] = np.nanmean(structured_data[i][j])

	data = pd.DataFrame.from_dict(structured_data, orient='index')

	f, ax = plt.subplots(figsize=(12, 10))

	sns.heatmap(data, annot=False, fmt="f", linewidths=.5, ax=ax, center=0)

	# ax.text(-1, 0, environmental[0], rotation=45, fontsize=10)
	# ax.text(-0.5, 0, environmental[1], rotation=45, fontsize=10)
	# ax.text(25, 26.5, model[0], rotation=-25, fontsize=10)
	# ax.text(25, 28, model[1], rotation=-25, fontsize=10)

	plt.xlabel('Model Uncertainties')
	plt.ylabel('Environmental Uncertainties')

	plt.yticks(rotation=00)
	plt.show()


def plot_learned(context):
	sns.set()
	# axis0: y - environmental
	# axis1: x - model

	y_delim, x_delim = ' ', '\n'

	environmental = [f.name for f in context.environmental_uncertainties()]
	model = [f.name for f in context.model_uncertainties()]

	r = range(context.bins + 1) if isinstance(context.bins, int) else range(context.bins.shape[0])
	x_values = list(itertools.product(r, repeat=len(model)))
	y_values = list(itertools.product(r, repeat=len(environmental)))

	x_values = [x_delim.join(str(xi) for xi in x) for x in x_values]
	y_values = [y_delim.join(str(yi) for yi in y) for y in y_values]

	x_dict = {x: [] for x in x_values}
	structured_data = {y: copy.deepcopy(x_dict) for y in y_values[::-1]}
	rules = context.processed_learned_data
	rules = sorted(rules)
	context_bins = list(context.bins)

	def get_bins_from_range(lb, ub):
		a = context_bins.index(lb)
		b = context_bins.index(ub)
		return list(range(a, b + 1))

	def bins_to_string_index(bins, d):
		b = [d.join(str(xi) for xi in x) for x in bins]
		return b

	for rule in rules:
		eu_bins, mu_bins = [], []

		for eu, _, (lb, ub) in rule.environmental_uncertainties:
			bins = get_bins_from_range(lb, ub)
			eu_bins.append(bins)

		for mu, _, (lb, ub) in rule.model_uncertainties:
			bins = get_bins_from_range(lb, ub)
			mu_bins.append(bins)

		i_indexes = list(itertools.product(*eu_bins))
		j_indexes = list(itertools.product(*mu_bins))

		i_indexes = bins_to_string_index(i_indexes, y_delim)
		j_indexes = bins_to_string_index(j_indexes, x_delim)

		for i_index in i_indexes:
			for j_index in j_indexes:
				structured_data[i_index][j_index].append((rule.outcome, rule.confidence, rule.experience))

	for i in structured_data:
		for j in structured_data[i]:
			if len(structured_data[i][j]) > 0:
				outcomes, confidences, experiences = zip(*structured_data[i][j])
				weights = [c * e for c, e in zip(confidences, experiences)]
				structured_data[i][j] = np.average(outcomes, weights=weights)
			else:
				structured_data[i][j] = np.nan

	data = pd.DataFrame.from_dict(structured_data, orient='index')
	f, ax = plt.subplots(figsize=(12, 10))
	sns.heatmap(data, annot=False, fmt="f", linewidths=.5, ax=ax, center=0)

	plt.xlabel('Model Uncertainties')
	plt.ylabel('Environmental Uncertainties')

	plt.yticks(rotation=00)
	plt.show()







