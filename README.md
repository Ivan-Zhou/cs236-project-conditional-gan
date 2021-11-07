
## Training
```
bash build.sh # build docker
bash launch.sh # launch docker
bash run.sh # start training
```

### To use MNIST dataset

```
./run.sh mnist
```

## Evaluation
```
# run evaluation with the given checkpoint path
./eval.sh PATH_TO_CHECKPOINT

# For example
./run_eval.sh default-project/out/test/ckpt/5000.pth
```