python train.py --dataroot ./datasets/face2emoji_small --name face2emoji_small --model cycle_gan
python test.py --dataroot datasets/face2emoji_small/ --name face2emoji_small --model cycle_gan

docker run --gpus=all -it --rm --volume ${PWD}:/workspace -w /workspace --name ft-work  -e WORKSPACE=/workspace nvcr.io/nvidia/pytorch:21.02-py3


python train.py --dataroot ./datasets/face2emoji --name face2emoji --model cycle_gan --gpu_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --batch_size 128 --num_threads 16 --norm instance --continue_train
