# Strategy Learning System 
#### The implementation of my M.S. Thesis titled: *Learning Explanatory Models for Robust Decision-Making Under Deep Uncertainty*

Additional documentation can be found on the [homepage](http://brodderick.com/thesis) for this project.

Requirements:
* [`requirements.txt`](https://github.com/brodderickrodriguez/strategy_learning_system/blob/master/requirements.txt)
* [NetLogo](https://ccl.northwestern.edu/netlogo/download.shtml) 6
* Python 3.7+

Installation:
* `git clone https://github.com/brodderickrodriguez/strategy_learning_system.git`
* `cd /path/to/strategy_learning_system`
* `python3 -m venv .env`
* `source .env/bin/activate`
* `pip3 install -r requirements.txt`

To experiment with the Contaminant Plume Model:
* All of the steps above
* `cd examples; git clone https://github.com/brodderickrodriguez/contaminant_plume_model.git`

The following code snippet is for the Contaminant Plume Model. It can be found in `./examples/plume_model.py`.

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




