# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Dec. 2019


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd


def plot_explored(context):
	sns.set()
	# axis0: y - environmental
	# axis1: x - model

	environmental = [f.name for f in context.environmental_uncertainties()]
	model = [f.name for f in context.model_uncertainties()]

	data = context.processed_exploratory_results

	x_values = list(itertools.product(range(context.bins.shape[0]), repeat=len(model)))
	y_values = list(itertools.product(range(context.bins.shape[0]), repeat=len(environmental)))

	x_values = [''.join(str(xi) for xi in x) for x in x_values]
	y_values = [''.join(str(yi) for yi in y) for y in y_values]

	x_values[0] = 'a'

	x_dict = {x: [] for x in x_values}
	structured_data = {y: x_dict for y in y_values[::-1]}

	def row_to_string(row, uncertainties):
		n = [int(i) for i in row[uncertainties]]
		n_idx = ''.join(str(i) for i in n)
		return n_idx

	for i in range(data.shape[0]):
		row = data.iloc[i]
		env_idx = row_to_string(row, environmental)
		mod_idx = row_to_string(row, model)
		structured_data[env_idx][mod_idx].append(row['rho'])

	for y_key in structured_data:
		y_i = structured_data[y_key]
		for x_key in y_i:
			y_i[x_key] = np.mean(y_i[x_key])

	data = pd.DataFrame.from_dict(structured_data, orient='index')
	f, ax = plt.subplots(figsize=(12, 10))
	sns.heatmap(data, annot=False, fmt="f", linewidths=.5, ax=ax)
	plt.show()


def plot_learned(medicator, context):
	pass
	# print('hi')e