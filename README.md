# Differentiable Mel spectrogram

The code in the 'main' branch is subject to change in the future, the exact code used in the paper is kept in a separate branch callend 'eusipco2023'.

This README.md explains how to reproduce the main results in the paper

    Under review at EUSIPCO 2023.
    
    TODO: WRITE ABOUT AND REFERENCE THE PAPER
        
# Clone repository and setup the environment
Clone the 'eusipco2023' branch of the repository and change working directory. All commands assume that this is the working directory.

    git clone -b eusipco2023 https://github.com/johnmartinsson/differentiable-mel-spectrogram.git
    cd differentiable-mel-spectrogram
    
Uing pip:
    
    pip install -r requirements.txt

Using Anaconda:

    conda create -n torchaudio
    conda activate torchaudio
    
    # solving these environments are unfortunately a bit slow... have patience
    conda install numpy tqdm seaborn pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
    conda install -c conda-forge ray-tune librosa
    
# Do everything in one script
Run the doit.sh script to download the audio data, run the experiments and produce the plots.

    sh doit.sh
    
which runs the commands

    # download the FSD dataset
    sh download_data.sh

    # run all the experiments (takes ~16h on a 2080Ti)
    sh run_experiments.sh

    # produce all the figures (takes ~5 min on a 2080Ti)
    python produce_figures.py --split=test --ray_root_dir=$(pwd)/ray_results/ --data_dir=$(pwd)/data/audio-mnist

The figures are stored in ./results/figures

    # figure 1 in paper
    data_example.pdf
    
    # figure 2 in paper
    test_time_frequency.pdf
    
    # figure 3 in paper
    test_audio_mnist.pdf

That should be all.

# Download the Free Spoken Digits dataset
The Free Spoken Digits dataset can be downloaded using:

    sh download_data.sh
    
which will download the Free Spoken Digits dataset using the following commands:

    # DOI: 10.5281/zenodo.1342401
    wget https://zenodo.org/record/1342401/files/Jakobovski/free-spoken-digit-dataset-v1.0.8.zip
    unzip free-spoken-digit-dataset-v1.0.8.zip
    mkdir data
    mv Jakobovski-free-spoken-digit-dataset-e9e1155/ data/audio-mnist
    rm free-spoken-digit-dataset-v1.0.8.zip

# Run experiments
The experiments will not reproduce the exact results in the paper, the random seed has never been fixed, but the same trends should of the averages and standard deviations should be observed when re-running the experiments.

Run all the experiments.

    sh run_experiments.sh

Only synthetic data.

    python main.py --num_samples=5 --max_epochs=10000 --name=time_frequency --ray_root_dir=$(pwd)/ray_results/ --data_dir=$(pwd)/data/audio-mnist
   
Only Free Spoken Digit dataset.

    python main.py --num_samples=5 --max_epochs=10000 --name=audio_mnist --ray_root_dir=$(pwd)/ray_results/ --data_dir=$(pwd)/data/audio-mnist
    
The code uses 0.25 GPUs and 2 CPUs per experiment, edit the tune.with_resources line in main.py if you want to use more or less GPUs or CPUs. Defaults to cuda:1 device.

# Produce figures

This re-produces all the figures in the paper.

Test data. This requires looping through all model configurations and making predictions on the test set, which takes a couple of minutes on the GPU. The DataFrame is stored in the ./results directory as well as the predictions and labels for each model, which are loaded the next time the script is run to prevent re-running all test predictions.

    # produces all figures in the paper
    python produce_figures.py --split=test --ray_root_dir=$(pwd)/ray_results/ --data_dir=$(pwd)/data/audio-mnist
    
Produces the figures used in the paper and puts them in the $(pwd)/results/figures directory.

# Explore the datasets and models

TODO: reference a notebook, and write a tutorial/guide in the notebook.
