
## Prepare the dataset
### Small emoji dataset from Kaggle
- [emoji data 4k](https://www.kaggle.com/mostafamozafari/bitmoji-faces/version/1): 4k emoji dataset
- [celebA](https://drive.google.com/file/d/1t-qDQQqJdX8B9ZcyO6YPqNVy8GTmILg9/view?usp=sharing): 203K celebrity face dataset
```
cd cycle_gan
mkdir -p face2emoji_small/trainA   # put celebA(real person) images here
mkdir -p face2emoji_small/trainB   # put emoji images
mkdir -p face2emoji_small/testA    # put celebA images, no duplicate from trainA
mkdir -p face2emoji_small/testA    # put emoji images, no duplicate from trainB
```

### Large emoji dataset generated from Bitmoji API
- [emoji data 900k](https://drive.google.com/file/d/1p3Y9kGKnPdo-Fu5CWFcsDWODSuLRPMrU/view?usp=sharing): 900k emoji dataset with properties in a csv file
- [celebA](https://drive.google.com/file/d/1t-qDQQqJdX8B9ZcyO6YPqNVy8GTmILg9/view?usp=sharing): 203K celebrity face dataset

```
cd cycle_gan
mkdir -p face2emoji_large/trainA   # put celebA(real person) images here
mkdir -p face2emoji_large/trainB   # put emoji images
mkdir -p face2emoji_large/testA    # put celebA images, no duplicate from trainA
mkdir -p face2emoji_large/testA    # put emoji images, no duplicate from trainB
```

### Start training
```
cd cycle_gan
docker run --gpus all --rm -v $PWD:/workspace -it nvcr.io/nvidia/pytorch:21.09-py3 /bin/bash
pip install -r requirements.txt

DATASET="face2emoji_large" # or "face2emoji_small"
python train.py --dataroot ./$DATASET --name $DATASET --model cycle_gan --use_wandb
python test.py --dataroot ./$DATASET --name $DATASET --model cycle_gan
```
