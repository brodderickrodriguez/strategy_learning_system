# Brodderick Rodriguez
# Auburn University - CSSE
# 24 Aug. 2019


# TODO: HEAT MAPS ETC
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import copy
import itertools

def plot_explored(mediator, context):
	sns.set()

	# axis0: y - environmental
	# axis1: x - model

	environmental = [f.name for f in context.environmental_uncertainties()]
	model = [f.name for f in context.model_uncertainties()]

	data = copy.deepcopy(context.processed_exploratory_results)


	# print(data[environmental])

	x_values = list(itertools.product(range(context.bins.shape[0]), repeat=len(model)))
	y_values = list(itertools.product(range(context.bins.shape[0]), repeat=len(environmental)))

	print(x_values)




	# print(environmental)

	print(context.bins.shape[0])


	exit()

	# print(context.processed_exploratory_results)


	# Load the example flights dataset and conver to long-form
	flights_long = sns.load_dataset("flights")
	flights = flights_long.pivot("month", "year", "passengers")

	print(type(flights))

	print(flights)

	# Draw a heatmap with the numeric values in each cell
	f, ax = plt.subplots(figsize=(9, 6))
	sns.heatmap(context.processed_exploratory_results, annot=False, fmt="f", linewidths=.5, ax=ax)

	plt.show()


def plot_learned(medicator, context):
	pass
	# print('hi')e