# download the FSD dataset
sh download_data.sh

# run all the experiments (takes ~16h on a 2080Ti)
sh run_experiments.sh

# produce all the figures (takes ~5 min on a 2080Ti)
python produce_figures.py --split=test --ray_root_dir=$(pwd)/ray_results/ --data_dir=$(pwd)/data/audio-mnist
