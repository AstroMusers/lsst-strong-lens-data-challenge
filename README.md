# lsst-strong-lens-data-challenge

Can a classifier distinguish strong lenses from non-lenses across HSC and simulated SLSim images? We use these image populations to train lens classifiers and prepare submissions for the LSST Strong Lens Data Challenge.

# Setup

From the checkout root, source `scripts/set_repository_path.sh` to set `LSST_STRONG_LENS_DATA_CHALLENGE_PATH` in the current shell. Repository-local inputs and figures can live in the ignored `data/` and `visuals/` directories. The challenge dataset may remain elsewhere; `config.yml` continues to select its location.

1. Set up conda environment

`conda env create -f environment.yml`

2. Create a configuration file

`cp template_config.yml config.yml`

3. Download the data challenge data from https://slchallenge.cbpf.br/challengedata and unzip. Note that depending on how you do this, it might take a while.

4. Set the `data_dir` attribute in your `config.yml` file to the path where you downloaded the data challenge data

5. Run the `organize_data.py` script to preprocess the data and produce .npz files in the format that Tensorflow is expecting

We normalize each finite image band to [0, 1], mapping constant bands to zero contrast.
The NPZ organizer retains its gamma correction with a target median of 0.2, while
the RGB organizer uses linear normalization. Empty or nonfinite images and
multiband examples with different lens IDs are reported and skipped. These
transformations do not preserve absolute flux scales.
Integer pixel offsets are computed before floating-point conversion so nearby
64-bit values retain their contrast even at large absolute pixel levels.

The RGB organizer creates missing `train` and `test` class directories, then
removes existing files directly inside their `1_lens` and `0_nonlens`
subdirectories before writing new images. Nested directories are retained.
JPEG names include the dataset, eight-digit source index, and lens ID, so
repeated IDs do not overwrite earlier images. Class-directory names are unchanged.

We pack the four image classes into one archive and use the same reader during
training. Training and validation use a fixed shuffled row partition, with only
the training order reshuffled each epoch. This row-level split does not enforce
grouping by physical lens system.

We report binary accuracy at a logit threshold of zero, equivalent to a lens
probability threshold of 0.5. Older accuracy histories used a logit threshold
of 0.5, equivalent to a lens probability of about 0.62. Recompute those metrics
with the current threshold before comparing results. The cross-entropy loss and
model outputs are unchanged.
