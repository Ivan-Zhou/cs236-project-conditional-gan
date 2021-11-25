DATASET=$1
MODEL=$2
DATA_DIR="./data/Images"

if [ -z $DATASET ]; then
    DATASET="mnist"
fi

if [ -z $MODEL ]; then
    MODEL="cgan"
fi

if [ $DATASET == "stanford_dogs_top_10" ]; then
    DATA_DIR="./data/stanford_dogs_top_10"
fi

if [ $DATASET == "bitmoji-4k" ]; then
    DATA_DIR="./data/bitmoji-4k"
fi

JOB_NAME=$MODEL"_"$DATASET

echo "Dataset: "$DATASET
echo "Model: "$MODEL
echo "Job Name: "$JOB_NAME

cd default-project
python download.py

python train.py --name $JOB_NAME --model $MODEL --resume --data_dir $DATA_DIR --dataset $DATASET
