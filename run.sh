DATASET=$1
MODEL=$2

if [ -z $DATASET ]; then
    DATASET="mnist"
fi

if [ -z $MODEL ]; then
    MODEL="cgan"
fi

JOB_NAME=$MODEL"_"$DATASET

echo "Dataset: "$DATASET
echo "Model: "$MODEL
echo "Job Name: "$JOB_NAME

cd default-project
python download.py

python train.py --name $JOB_NAME --model $MODEL --resume --data_dir ./data/Images --dataset $DATASET
