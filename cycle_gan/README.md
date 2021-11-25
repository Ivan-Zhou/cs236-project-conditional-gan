
### Prepare the dataset
cd cycle_gan
mkdir -p face2emoji_small/trainA   # put celebA(real person) images
mkdir -p face2emoji_small/trainB   # put emoji images
mkdir -p face2emoji_small/testA    # put celebA images, no duplicate from trainA
mkdir -p face2emoji_small/testA    # put emoji images, no duplicate from trainB


### Start training
```
docker run --gpus all --rm -it nvcr.io/nvidia/pytorch:21.09-py3 /bin/bash
cd cycle_gan
pip install -r requirements.txt
python train.py --dataroot ./datasets/face2emoji_small --name face2emoji_small --model cycle_gan
python test.py --dataroot ./datasets/face2emoji_small --name face2emoji_small --model cycle_gan
```