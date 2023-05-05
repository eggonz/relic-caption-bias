# Code

The source code has been taken from [lick-caption-bias GitHub repo](https://github.com/rebnej/lick-caption-bias).

## Structure

Files from original repo ([modifications](#modifications) might have been made):

- `model.py`: despite the model being called `GenderClassifier`, it is a binary classifier used for all protected attributes (gender, race, age)


- `bias_dataset.py`: gender dataset
- `lstm_leakage.py`: gender lstm
- `bert_leakage.py`: gender bert


- `race_dataset.py`: race dataset
- `race_lstm_leakage.py`: race lstm
- `race_bert_leakage.py`: race bert

New files for age (copied from gender/race versions and adapted for age):

- `age_dataset.py`: age dataset
- `age_lstm_leakage.py`: age lstm
- `age_bert_leakage.py`: age bert

## Modifications

The code has been debugged and slightly modified to be able to run the experiments.
These are the most relevant modifications that were needed:

1. **Dependencies**
        
    Some dependencies had to be added for the code to actually work.
    There were also some compatibility between versions that failed at runtime, so they had to be adjusted.
    The files `env-LSTM.yml` and `env-BERT.yml` (in root of repo) contain the configurations we used that worked for our specific setting.
    
    We added `torchvision` and modified the versions of `pytorch` and `sentencepiece`.

    
2. **Download this once, then forget about it**
    
    We found that the code needed some previous action to be taken before using the `spacy` module.
    Before running the code, run the following command in the terminal:
    ```bash
    python -m spacy download en_core_web_sm
   # You can now load the package via spacy.load('en_core_web_sm')
    ```
    This action only needs to be done once. Once downloaded, the package can be used in the code.


3. **The forgotten *Windows***

    We found that there was a small mistake assuming the OS in which the code is run is Unix-based.
    When writing the files `train.csv` and `test.csv`, the code does not specify how to write linebreaks, which changes across OSs.
    
    We specified the writing dialect to be consistent across OSs.
    