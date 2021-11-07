cd default-project
python download.py
python train.py --name test --resume --data_dir ./data/Images
# cgan
python train.py --name debug_cgan --model cgan --resume --data_dir ./data/Images

# eval cgan
python eval_cgan.py --ckpt_path out/debug_cgan_2/ckpt/150000.pth --im_size 32