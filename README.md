# Dynamic Piggyback GAN by Pytorch.

본 repository는 Piggyback GAN Pytorch를 clone하여 2021 fall [AI502] Deep learning 과목의 term project를 진행하기 위한 실험 코드를 정리한 것입니다.

기존 Piggyback GAN source code에서 수정한 사항은 아래와 같습니다.

1. Piggyback GAN을 reproduce하는 과정에서 define된 generator를 nn.conv, nn.transposeconv를 각각에 대응되는 piggyback layer로 치환하는 과정에서 생기는 버그 수정.
2. CycleGAN을 사용한 unpaired image-to-image translation task에서 Pix2Pix 모델을 사용한 paired image-to-image translation task로 수정.
3. Dynamic lambda 구현을 위해 lambda를 task, layer마다 변경 가능하도록 수정.


# Piggyback GAN Pytorch

[Piggyback GAN](https://www.sfu.ca/~mnawhal/projects/zhai_eccv20.pdf) is a framework for lifelong learning in generative models. Specifically, it considers the problem of image-to-image translation using the CycleGAN and Pix2Pix framework. The goal (as with any lifelong learning framework) is to be able to learn as many tasks as possible, with minimal increase in no. of parameters. 

The CycleGAN and Pix2Pix code is mostly taken from [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 

The main contribution of this repository is the implementation of PiggybackConv and PiggybackTransposeConv module in ```./models/networks.py```. These are custom convolution modules that have unconstrained filters and piggyback filters. As described in the paper, there are only unconstrained filters for Task 1. In the subsequent tasks, there are both piggyback and unconstrained filters. An illustartion of this task-wise filter learning is shown below with a figure from the paper: ![](./README_figures/pb_gan_pic.png)

The repository also implements GPU parallelism through nn.DistributedDataParallel. 

## Results
The performance of Piggyback GAN on each of the sequential tasks at the end of 4 tasks is visualized below: 

![](./README_figures/pb_gan_all_tasks.png)

## Instructions to run
First, run the following to setup the environment: 
```
conda env create -f environment.yml
```

Download 4 cycleGAN datasets:
```
bash ./datasets/download_cyclegan_dataset.sh maps
bash ./datasets/download_cyclegan_dataset.sh facades
bash ./datasets/download_cyclegan_dataset.sh vangogh2photo
```
For cityscapes, read instructions on how to download and prepare, from: ```./datasets/prepare_cityscapes_dataset.py```

To perform training, run: 
```
python pb_cycleGAN.py train=True
```

After training, the following folder structure is created: 
```
+-- checkpoints
    +-- Task_1_cityscapes_cycleGAN
        +-- Intermediate_train_images
        +-- filters.pt 
        +-- latest_checkpoint.pt
    +-- Task_2_maps_cycleGAN
        +-- Intermediate_train_images
        +-- filters.pt 
        +-- latest_checkpoint.pt
    +-- Task_3_facades_cycleGAN
        +-- Intermediate_train_images
        +-- filters.pt 
        +-- latest_checkpoint.pt
    +-- Task_4_vangogh2photo_cycleGAN
        +-- Intermediate_train_images
        +-- filters.pt 
        +-- latest_checkpoint.pt
```
To perform testing from trained model, use:
```
python pb_cycleGAN.py train=False
```

During the testing phase, the code restores the filters from the last task and uses only parts of it for every task. 
This is because with every task, the weights (unconstrined filter bank and piggyback weight matrix) of current and all previous
tasks are stored. 

After testing code is run, a folder called Test_images gets created under each Task_x_y_cycleGAN folder.

## Todo: 
- [x] Add results.
- [x] Include hydra for config management. 
- [x] Add dataset download scripts.
- [ ] Include experiemnts on pix2pix
- [ ] Calculate FID and tabulate results.
