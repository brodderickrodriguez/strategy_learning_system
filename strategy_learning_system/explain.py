# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Dec. 2019


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import copy


# axis0: y - environmental
# axis1: x - model
_y_bin_delim = ' '
_x_bin_delim = '\n'


def _get_axis_labels(context):
	environmental = [f.name for f in context.environmental_uncertainties()]
	model = [f.name for f in context.model_uncertainties()]

	r = range(context.bins.shape[0])
	x_values = list(itertools.product(r, repeat=len(model)))
	y_values = list(itertools.product(r, repeat=len(environmental)))

	x_values = [_x_bin_delim.join(str(xi) for xi in x) for x in x_values]
	y_values = [_y_bin_delim.join(str(yi) for yi in y) for y in y_values]

	return x_values, y_values, model, environmental


def _make_heat_map(structured_data):
	sns.set()
	data = pd.DataFrame.from_dict(structured_data, orient='index')
	f, ax = plt.subplots(figsize=(12, 10))
	sns.heatmap(data, annot=False, fmt="f", linewidths=.5, ax=ax)

	plt.xlabel('Model Uncertainties')
	plt.ylabel('Environmental Uncertainties')

	plt.yticks(rotation=00)
	plt.show()


def plot_explored(context):
	x_values, y_values, model, environmental = _get_axis_labels(context)
	x_dict = {x: [] for x in x_values}
	structured_data = {y: copy.deepcopy(x_dict) for y in y_values[::-1]}
	data = context.processed_exploratory_results

	def uncertainties_to_string(uncertainties, delim):
		n = [int(n) for n in uncertainties]
		s = delim.join(str(ni) for ni in n)
		return s

	for i in range(data.shape[0]):
		row = data.iloc[i]
		environmental_i = list(row[environmental])
		model_i = list(row[model])
		outcome = row['rho']

		data_i_idx = uncertainties_to_string(environmental_i, _y_bin_delim)
		data_j_idx = uncertainties_to_string(model_i, _x_bin_delim)
		structured_data[data_i_idx][data_j_idx].append(outcome)

	for i in structured_data:
		for j in structured_data[i]:
			structured_data[i][j] = np.nanmean(structured_data[i][j])

	_make_heat_map(structured_data)


def _get_bins_from_lb_ub(bins, lb, ub):
	def _get(x):
		try:
			return bins.index(x)
		except ValueError:
			# check if left bin is closer in value than the right bin
			for i in range(1, len(bins)):
				if bins[i] > x:
					if np.abs(bins[i - 1] - x) < np.abs(bins[i] - x):
						return i - 1
					else:
						return i

	a = _get(lb)
	b = _get(ub)
	return list(range(a, b + 1))


def plot_learned(context):
	x_values, y_values, model, environmental = _get_axis_labels(context)
	x_dict = {x: [] for x in x_values}
	structured_data = {y: copy.deepcopy(x_dict) for y in y_values[::-1]}
	rules = sorted(context.processed_learned_data)[::-1]
	context_bins = list(context.bins)

	for rule in rules:
		x_bins, y_bins = [], []

		for eu, _, (lb, ub) in rule.environmental_uncertainties:
			y_bins_i = _get_bins_from_lb_ub(context_bins, lb, ub)
			y_bins.append(y_bins_i)

		for mu, _, (lb, ub) in rule.model_uncertainties:
			x_bins_i = _get_bins_from_lb_ub(context_bins, lb, ub)
			x_bins.append(x_bins_i)

		y_indexes = list(itertools.product(*y_bins))
		x_indexes = list(itertools.product(*x_bins))
		y_indexes = [_y_bin_delim.join(str(yi) for yi in y) for y in y_indexes]
		x_indexes = [_x_bin_delim.join(str(xi) for xi in x) for x in x_indexes]

		for y_idx in y_indexes:
			for x_idx in x_indexes:
				y_x_val = rule.outcome, rule.confidence, rule.experience
				structured_data[y_idx][x_idx].append(y_x_val)

	for y_key in structured_data:
		for x_key in structured_data[y_key]:
			y_x_val = np.nan
			y_x_data = structured_data[y_key][x_key]

			if len(y_x_data) > 0:
				outcomes, confidences, experiences = zip(*structured_data[y_key][x_key])
				confidence_weight = [c for c, e in zip(confidences, experiences)]
				y_x_val = np.average(outcomes)

			structured_data[y_key][x_key] = y_x_val

	_make_heat_map(structured_data)
