
DATASET=$1

# Local and Gru IO
if [ -z $DATASET ]; then
    DATASET="local"
fi

echo "Dataset: "$DATASET
cd default-project
python download.py
python train.py --name test --resume --data_dir ./data/Images --dataset $DATASET
