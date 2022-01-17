# StyleGan Based on Jittor

A [Jittor](https://cg.cs.tsinghua.edu.cn/jittor) implementation of StyleGan.



## Environment

python 3.7

jittor 1.3.1.33

CUDA 11.0

## usage

### dataset

download color_symbol_7k dataset from http://jittor40.randonl.me/  and unzip it into the root path of this project.

### training

```shell
python train.py
```

the training samples and checkpoints will be saved in ./train_output

### inference

1. choose a checkpoint and put it in ./checkpoint/generator.pkl

2. you can get a sample generated from a random vector, using

   ```
   python generate.py
   ```

3. you can generate latent interpolation result images at ./interp using

   ```
   python interp.py
   ```

4. you can generate a video demo at ./demo.mp4 using

   ```
   python video.py
   ```