python train.py --dataroot ./datasets/face2emoji_small --name face2emoji_small --model cycle_gan
python test.py --dataroot datasets/face2emoji_small/ --name face2emoji_small --model cycle_gan

docker run --gpus=all -it --rm --volume ${PWD}:/workspace -w /workspace --name ft-work  -e WORKSPACE=/workspace nvcr.io/nvidia/pytorch:21.02-py3