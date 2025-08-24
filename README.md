# lsst-strong-lens-data-challenge

# Setup

1. Set up conda environment

`conda env create -f environment.yml`

2. Create a configuration file

`cp template_config.yml config.yml`

3. Download the data challenge data from https://slchallenge.cbpf.br/challengedata and unzip. Note that depending on how you do this, it might take a while.

4. Set the `data_dir` attribute in your `config.yml` file to the path where you downloaded the data challenge data

5. Run the `organize_data.py` script to preprocess the data and produce .npz files in the format that Tensorflow is expecting