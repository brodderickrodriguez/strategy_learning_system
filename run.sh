
echo "y" | pip3 uninstall strategy_learning_system

pip3 install .

clear

python3 case_studies/prey_pred_model/prey_pred_model.py

rm -rf tmp*
