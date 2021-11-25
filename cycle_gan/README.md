


```
docker run --gpus all --rm -it nvcr.io/nvidia/pytorch:21.09-py3 /bin/bash
pip install -r requirements.txt
python train.py --dataroot ./datasets/face2emoji_small --name face2emoji_small --model cycle_gan
python test.py --dataroot ./datasets/face2emoji_small --name face2emoji_small --model cycle_gan
```