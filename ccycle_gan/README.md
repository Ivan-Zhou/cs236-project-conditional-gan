
## Prepare the dataset
- [emoji data 900k](https://drive.google.com/file/d/1p3Y9kGKnPdo-Fu5CWFcsDWODSuLRPMrU/view?usp=sharing): 900k emoji dataset
- [emoji attributes csv](https://drive.google.com/file/d/1AnjbbxYxXq1Kyl9iA-GnqUVIuoym46Dq/view?usp=sharing): processed attributes
- [celebA](https://drive.google.com/file/d/1t-qDQQqJdX8B9ZcyO6YPqNVy8GTmILg9/view?usp=sharing): 203K celebrity face dataset
- [celebA attributes csv](https://drive.google.com/file/d/1uc7tBKF6BN1JUX7BsAXpQCRYBHLPtnwo/view?usp=sharing): processed attributes for celebA

```
cd ccycle_gan
python util/prepare_data.py
```

### Start training
```
cd ccycle_gan
docker run --gpus all --rm -v $PWD:/workspace -it nvcr.io/nvidia/pytorch:21.09-py3 /bin/bash
pip install -r requirements.txt

DATASET="face2emoji_conditional_5k"
python train.py --dataroot ./$DATASET --name $DATASET --model ccycle_gan # --use_wandb
python test.py --dataroot ./$DATASET --name $DATASET --model ccycle_gan
```
