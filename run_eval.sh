CKPT=$1
PROJECT_DIR=default-project
python $PROJECT_DIR/download.py
python $PROJECT_DIR/eval.py \
    --data_dir $PROJECT_DIR/data/Images \
    --ckpt_path $CKPT \
    --im_size 32
