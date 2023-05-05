# Scripts

This folder contain the scripts that reproduce our experiments.

There is a file for each of these configurations:

- gender + LSTM
- gender + BERT-finetuned
- gender + BERT-pretrained
- race + LSTM
- race + BERT-finetuned
- race + BERT-pretrained
- age + LSTM
- age + BERT-finetuned
- age + BERT-pretrained

Each script runs one of the previous configurations across all selected models and seeds:

- Models(9): NIC, SAT, FC, Att2in, UpDn, Transformers, OSCAR, NIC+, NIC+Equalizer
- Seeds(10): 0, 12, 456, 789, 100, 200, 300, 400, 500, 1234

The scripts can be run from the command line. Here's an example:

```bash
sh run_gender_lstm.sh path/to/data
```

The data directory argument is optional.

*NOTE: running BERT experiments requires a GPU to be available.*
