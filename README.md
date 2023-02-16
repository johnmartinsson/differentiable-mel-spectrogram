TODO: Update code and replace sigma with lambda in variable names. To be consistent with article and avoid confusion.

# Differentiable time-frequency transforms

This README.md explains how to reproduce the main results in the paper

    TODO: WRITE ABOUT AND REFERENCE THE PAPER

# Setup environment

    conda install requirements.txt

# Run experiments

    python main.py --num_samples=4000 --max_epochs=500 --name=dtf_sigma_search
    
The code uses 0.25 GPUs and 2 CPUs per experiment, edit the tune.with_resources line in main.py if you want to use more or less GPUs or CPUs.

# Produce figures

    python produce_figures.py --name=dtf_sigma_search
    
Produces the figures used in the paper and puts them in the ./figures directory.

# Explore the datasets

TODO: reference the notebook, and write a tutorial/guide in the notebooks.

Explore the results

TODO: reference the notebook, and explain how to look at other results and produce other plots.
