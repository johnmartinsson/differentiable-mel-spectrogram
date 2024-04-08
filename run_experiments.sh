CUDA_VISIBLE_DEVICES=0 python main.py --num_samples=1 --max_epochs=1000 --name=audio_mnist --ray_root_dir=$(pwd)/ray_results --data_dir=$(pwd)/data/audio_mnist
CUDA_VISIBLE_DEVICES=0 python main.py --num_samples=1 --max_epochs=1000 --name=time_frequency --ray_root_dir=$(pwd)/ray_results --data_dir=$(pwd)/data/time_frequency
#CUDA_VISIBLE_DEVICES=0 python main.py --num_samples=2 --max_epochs=1000 --name=esc50 --ray_root_dir=$(pwd)/ray_results --data_dir=$(pwd)/data/esc50
