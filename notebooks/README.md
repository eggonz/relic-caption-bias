# Notebooks

These notebooks provide useful tools and also define a pipeline for the preprocessing of the data.

The notebooks were used for the experiments and the process can be reproduced executing the notebooks in order.

1. `data2csv`: util - see data as `csv` instead of `pkl`
2. `age_models_intersection`: data preprocessing - select image IDs used in SAT, OSCAR, NIC+ and NIC+Equalizer
3. `prepare_coco`: util - select images from COCO that are used for age (useful for hand annotation)
4. `combine_hand_annot`: data preprocessing - combine hand annotations from different people
5. `prepare_hand_annot`: data preprocessing - reformat the hand-annotated data to match the code requirements
6. `create_age_dataset`: data preprocessing - create `csv` dataset for age, containing captions and labels
7. `csv_pkl_conversion`: data preprocessing - `csv2pkl` and `pkl2csv` tools. Converts age dataset `csv` to `pkl`

Ideally, the notebooks should be run in order. Utils might be skipped.

## Preparing data for `runnotate`

The notebook `3_prepare_coco.ipynb` prepares the COCO images for hand annotation.
[Download the COCO 2014 datasets](https://cocodataset.org/#download) from the official page and place them under `res/mscoco/`.
Follow the instructions in notebook 3 and run all the cells.
An output folder containing all the images will be generated.
The names of those images have the format `<image-id>.jpg`.
This folder should be the data folder provided to the `runnotate` tool.
