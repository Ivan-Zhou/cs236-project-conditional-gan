
## Training
### Set up and launch docker
```
bash build.sh # build docker
bash launch.sh # launch docker
```

To launch a training job, please pass two parameters:
- `DATASET`: The name of the Dataset for training (`stanford_dog` or `mnist`)
- `MODEL`: The type of the model for training (`default` or `cgan`)

For example:
```
# Train CGAN on MNIST
./run.sh mnist cgan

# Train CGAN on Stanford Dog
./run.sh stanford_dog cgan
./run.sh stanford_dogs_top_10 cgan

# Train CGAN on Fashion-MNIST
./run.sh fashion-mnist cgan
```

## Evaluation
To launch an eval job, please pass three parameters:
- `DATASET`: The name of the Dataset for evaluation (`stanford_dog` or `mnist`)
- `MODEL`: The type of the model for evaluation (`default` or `cgan`)
- `EXP_DIR`: The directory that contains checkpoints. The output sample png and metrics json will also be saved there

For example:
```
./run_eval.sh stanford_dog cgan default-project/out/cgan_stanford_dog
```
