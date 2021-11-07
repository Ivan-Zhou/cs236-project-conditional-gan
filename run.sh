
DATASET=$1

# Local and Gru IO
if [ -z $DATASET ]; then
    DATASET="local"
fi

echo "Dataset: "$DATASET
cd default-project
python download.py
python train.py --name test --resume --data_dir ./data/Images --dataset $DATASET


#######################
# cgan
python train.py --name debug_cgan_2 --model cgan --resume --data_dir ./data/Images --max_steps 9999999

# eval cgan
python eval_cgan.py --ckpt_path out/debug_cgan_2/ckpt/150000.pth --im_size 32

python train.py --name debug_cgan_mnist --model cgan --resume --dataset mnist --max_steps 9999999 --num_classes 10
