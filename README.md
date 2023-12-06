#  **RELIC: Reproducibility and Extension on the LIC metric’s performance in quantifying bias in captioning models**

*University of Amsterdam, 2023. NeurIPS 2023.*

Paula Antequera, Egoitz Gonzalez, Marta Grasa and Martijn van Raaphorst

[[paper]](assets/relic_paper.pdf?raw=true) | [[poster]](assets/poster.pdf?raw=true) | [[slides]](assets/slides.pdf?raw=true) | [[OpenReview]](https://openreview.net/forum?id=9_hCoP3LXwy)

This repository contains source code used in "**RELIC: Reproducibility and Extension on the LIC metric’s performance in quantifying bias in captioning models**", a reproducibility study on the paper [Quantifying Societal Bias Amplification in Image Captioning](https://openaccess.thecvf.com/content/CVPR2022/html/Hirota_Quantifying_Societal_Bias_Amplification_in_Image_Captioning_CVPR_2022_paper.html) (CVPR 2022, Oral), whose [source code](https://github.com/rebnej/lick-caption-bias.git) was used as a base for the code in this repository.
An overview of their work can be found in this [website](https://sites.google.com/view/cvpr-2022-quantify-bias/home).

## Repository structure

`code/` - our source code for the python project

`scripts/` - python scripts to run the experiments

`notebooks/` - our python notebooks to create the age dataset

`data/` - data for the experiments

`res/` - relevant resources

`assets/` - additional material

## Setup

Two environments are provided, containing the necessary dependencies to run the code.
Depending on the experiment, *LSTM* or *BERT* environment should be used.

Create environment for *LSTM*:

```bash
conda env create -f env-LSTM.yml
```

Create environment for *BERT*:

```bash
conda env create -f env-BERT.yml
```

Activate conda environment for *LSTM*:

```bash
conda activate lstm
```

Activate conda environment for *BERT*:

```bash
conda activate bert
```

### Additional packages

To be able to use the `en_core_web_sm` from `spacy` in the code, it has to be downloaded beforehand with the following command:

```bash
python -m spacy download en_core_web_sm
```

More information about the code can be found at `code/README.md`.

### Data

The gender and race experiments require the same data that the authors from the original paper used, which can be downloaded from [here](https://drive.google.com/drive/folders/1PI03BqcnhdXZi2QY9PUHzWn4cxgdonT-).
Download the `bias_data/` folder and place it under `data/` in the root of the repository.

### Age experiments

Running age experiments might require additional data and preprocessing it.
We provide some python notebooks, under `notebooks/`, to do so.
These notebooks also make use of `data/bias_data/` previously downloaded.
For this, follow the instructions in `notebooks/README.md`.
The notebooks use the previously created [hand annotations](#hand-annotations).

## Running the code

Run a script with the experiments and store the logs into an output file:

```bash
sh run_gender_lstm_model.sh &> lstm_output.out &
```

More information about the scripts can be found at `scripts/README.md`.

## Hand annotations

The hand-annotated data for age were made using the `runnotate` tool, which code is provided under `res/` folder.
`runnotate` requires the dataset of COCO images and produces an output with the hand annotations.
Example outputs are provided under `res/annotations/`, which are the ones that were used to run the experiments.
We provide a notebook to generate the data to do the hand annotations, see `notebooks/README.md` for more information.
