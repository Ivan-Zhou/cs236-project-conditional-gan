DATASET=$1
MODEL=$2
EXP_DIR=$3

if [ -z $DATASET ]; then
    DATASET="mnist"
fi

if [ -z $MODEL ]; then
    MODEL="cgan"
fi

if [ -z $EXP_DIR ]; then
    EXP_DIR=$MODEL"_"$DATASET
fi

PROJECT_DIR=default-project
python $PROJECT_DIR/download.py
python $PROJECT_DIR/eval.py \
    --data_dir $PROJECT_DIR/data/Images \
    --dataset $DATASET \
    --model $MODEL \
    --out_dir $EXP_DIR \
    --exp_dir $EXP_DIR \
    --im_size 32
