CUDA_VISIBLE_DEVICES=1 python main.py --num_samples=5 --max_epochs=1000 --name=audio_mnist --ray_root_dir=./ray_results --data_dir=$(pwd)/data/audio-mnist
CUDA_VISIBLE_DEVICES=1 python main.py --num_samples=5 --max_epochs=1000 --name=time_frequency --ray_root_dir=./ray_results --data_dir=$(pwd)/data/audio-mnist
