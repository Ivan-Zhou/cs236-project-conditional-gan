cd default-project
python download.py
python train.py --name test --resume --data_dir ./data/Images
# cgan
python train.py --name debug_cgan --model cgan --resume --data_dir ./data/Images