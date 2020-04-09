# copy and paste this into your Unix-based Terminal!

# install this repo
git clone https://github.com/brodderickrodriguez/strategy_learning_system.git

# set up virtual environment 
cd strategy_learning_system
python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt

# set up contaminant plume experiment
cd examples
mkdir experiment_data
git clone https://github.com/brodderickrodriguez/contaminant_plume_model.git
