TODO: Update code and replace sigma with lambda in variable names. To be consistent with article and avoid confusion.

# Differentiable time-frequency transforms

This README.md explains how to reproduce the main results in the paper

    TODO: WRITE ABOUT AND REFERENCE THE PAPER

# Download AudioMNIST
    
    # DOI: 10.5281/zenodo.1342401
    wget https://zenodo.org/record/1342401/files/Jakobovski/free-spoken-digit-dataset-v1.0.8.zip
    unzip free-spoken-digit-dataset-v1.0.8.zip
    mkdir data
    mv Jakobovski-free-spoken-digit-dataset-e9e1155/ data/audio-mnist
    
Edit the "audio_mnist" method in the file search_spaces.py, and update the 'source_dir' variable to point to the ahsolute data/audio-mnist directory.

# Setup environment

    conda install requirements.txt

# Run experiments

All experiments.

    sh run_experiments.sh

Only synthetic data.

    python main.py --num_samples=5 --max_epochs=10000 --name=time_frequency --ray_root_dir=./ray_results/
   
Only Free Spoken Digit dataset.

    python main.py --num_samples=5 --max_epochs=10000 --name=audio_mnist --ray_root_dir=./ray_results/
    
The code uses 0.25 GPUs and 2 CPUs per experiment, edit the tune.with_resources line in main.py if you want to use more or less GPUs or CPUs. Defaults to cuda:1 device.

# Produce figures

Validation data.

    python produce_figures.py --name=time_frequency --split=valid --ray_root_dir=./ray_results/
    python produce_figures.py --name=audio_mnist --split=valid --ray_root_dir=./ray_results/

Test data. This requires looping through all model configurations and making predictions on the test set, which takes a couple of minutes on the GPU. The DataFrame is stored in the ./results directory as well as the predictions and labels for each model, which are loaded the next time the script is run to prevent re-running all test predictions.

    # figure 2
    python produce_figures.py --name=time_frequency --split=test --ray_root_dir=./ray_results/
    
    # figure 3
    python produce_figures.py --name=audio_mnist --split=test --ray_root_dir=./ray_results/
    
Produces the figures used in the paper and puts them in the ./results/figures directory.

# Explore the datasets

TODO: reference the notebook, and write a tutorial/guide in the notebooks.

Explore the results

TODO: reference the notebook, and explain how to look at other results and produce other plots.
