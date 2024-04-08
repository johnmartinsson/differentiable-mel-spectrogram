# download the FSD dataset
sh download_data.sh

# run all the experiments (takes ~16h on a 2080Ti)
sh run_experiments.sh

# run all test predictions
sh run_test_predictions.sh

# produce all the tables
python produce_tables.py --ray_results_dir=$(pwd)/ray_results/
