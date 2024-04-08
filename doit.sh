# download the FSD dataset
echo "downloading all data ..."
sh download_data.sh

# run all the experiments (takes ~16h on a 2080Ti)
echo "running all experiments ..."
sh run_experiments.sh

# run all test predictions
echo "running all test predictions ..."
sh run_test_predictions.sh

# produce all the tables
echo "producing all tables ..."
python produce_tables.py --ray_results_dir=$(pwd)/ray_results/
