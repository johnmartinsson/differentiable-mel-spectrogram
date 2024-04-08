# download the FSD dataset
sh download_data.sh

# run all the experiments (takes ~16h on a 2080Ti)
sh run_experiments.sh

# produce all the figures (takes ~5 min on a 2080Ti)
python produce_tables.py
