# setup

BASEDIR="$(dirname "$0")"
REPO_DIR="$(pwd)/$BASEDIR/.."
WORKDIR="$REPO_DIR/code"

if [ -z "$1" ]
    then DATA_DIR="$REPO_DIR/data/bias_data"
else
    DATA_DIR="$(pwd)/$1"
fi

cd $WORKDIR

# experiments

python3 race_bert_leakage.py --seed 0 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 12 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 456 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 789 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 100 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 200 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 300 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 400 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 500 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 1234 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic --data_dir $DATA_DIR

python3 race_bert_leakage.py --seed 0 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model sat --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 12 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model sat --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 456 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model sat --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 789 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model sat --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 100 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model sat --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 200 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model sat --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 300 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model sat --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 400 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model sat --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 500 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model sat --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 1234 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model sat --data_dir $DATA_DIR

python3 race_bert_leakage.py --seed 0 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model fc --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 12 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model fc --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 456 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model fc --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 789 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model fc --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 100 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model fc --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 200 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model fc --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 300 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model fc --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 400 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model fc --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 500 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model fc --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 1234 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model fc --data_dir $DATA_DIR

python3 race_bert_leakage.py --seed 0 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model att2in --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 12 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model att2in --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 456 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model att2in --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 789 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model att2in --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 100 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model att2in --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 200 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model att2in --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 300 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model att2in --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 400 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model att2in --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 500 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model att2in --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 1234 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model att2in --data_dir $DATA_DIR

python3 race_bert_leakage.py --seed 0 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model updn --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 12 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model updn --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 456 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model updn --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 789 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model updn --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 100 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model updn --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 200 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model updn --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 300 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model updn --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 400 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model updn --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 500 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model updn --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 1234 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model updn --data_dir $DATA_DIR

python3 race_bert_leakage.py --seed 0 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model transformer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 12 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model transformer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 456 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model transformer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 789 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model transformer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 100 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model transformer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 200 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model transformer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 300 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model transformer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 400 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model transformer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 500 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model transformer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 1234 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model transformer --data_dir $DATA_DIR

python3 race_bert_leakage.py --seed 0 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model oscar --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 12 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model oscar --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 456 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model oscar --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 789 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model oscar --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 100 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model oscar --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 200 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model oscar --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 300 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model oscar --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 400 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model oscar --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 500 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model oscar --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 1234 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model oscar --data_dir $DATA_DIR

python3 race_bert_leakage.py --seed 0 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_plus --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 12 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_plus --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 456 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_plus --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 789 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_plus --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 100 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_plus --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 200 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_plus --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 300 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_plus --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 400 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_plus --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 500 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_plus --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 1234 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_plus --data_dir $DATA_DIR

python3 race_bert_leakage.py --seed 0 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_equalizer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 12 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_equalizer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 456 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_equalizer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 789 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_equalizer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 100 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_equalizer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 200 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_equalizer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 300 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_equalizer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 400 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_equalizer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 500 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_equalizer --data_dir $DATA_DIR
python3 race_bert_leakage.py --seed 1234 --freeze_bert True --calc_model_leak True --calc_ann_leak True --cap_model nic_equalizer --data_dir $DATA_DIR
