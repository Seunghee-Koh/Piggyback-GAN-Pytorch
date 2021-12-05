# %%
import torch
import torch.nn as nn
import numpy as np
from configs.pix2pix_GAN_config import Pix2PixGANOptions
import torch.multiprocessing as mp
import torch.distributed as dist
from dataloaders.dataloader import AlignedDataset
import torch.utils
import os
from models.pix2pix_model import Pix2PixModel
import time
from models.networks import PiggybackConv, PiggybackTransposeConv, load_pb_conv, make_filter_list
import copy 
import sys
from pytorch_model_summary import summary as summary_

import pdb

# %%
def train(gpu, opt):

    device = torch.device('cuda:{}'.format(gpu)) if gpu>=0 else torch.device('cpu')
    if gpu >= 0:
        torch.cuda.set_device(gpu)
        rank = opt.nr * len(opt.gpu_ids) + gpu	                          
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://',                                   
            world_size=opt.world_size,                              
            rank=rank                                               
        )           
        # model = CycleGAN(opt, device)
        model = Pix2PixModel(opt, device)
        model = model.to(device)
 
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        if opt.train_continue:
            state_dict = torch.load(opt.ckpt_save_path+'/latest_checkpoint.pt')  
            model.load_state_dict(state_dict['model'])
            opt.start_epoch = state_dict['epoch']
            print(f'loaded {opt.start_epoch} epoch')

        train_dataset = AlignedDataset(opt)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler)       

    for epoch in range(opt.start_epoch, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        train_sampler.set_epoch(epoch)
        print("Length of loader is ",len(train_loader))
        for i, data in enumerate(train_loader):
            model.module.set_input(data)
            model.module.optimize_parameters() 
            if (i+1) % 50 == 0 and gpu<=0:
                print_str = (
                    f"Task: {opt.task_num} | Epoch: {epoch} | Iter: {i+1} | G: {model.module.loss_G:.5f} | "
                    f"G_L1: {model.module.loss_G_L1:.5f} | G_GAN: {model.module.loss_G_GAN:.5f} | "
                    f"D: {model.module.loss_D:.5f}"
                )
                print(print_str)


        model.module.update_learning_rate()  

        if gpu<=0:
            model.module.save_train_images(epoch)
            save_dict = {'model': model.state_dict(),
                        'epoch': epoch 
                    }
            torch.save(save_dict, opt.ckpt_save_path+'/latest_checkpoint.pt')
        dist.barrier()

    if gpu <= 0:
                   
        make_filter_list(model.module.netG, opt.netG_filter_list, opt.weights, opt.task_num)

        savedict_task = {'netG_filter_list':opt.netG_filter_list,
                            'weights':opt.weights
                        }
        torch.save(savedict_task, opt.ckpt_save_path+'/filters.pt')
        print(f'dict saved')

        # del netG_A_layer_list
        # del netG_B_layer_list
        del opt.netG_filter_list
        del opt.weights
        
    dist.barrier()
    del model

    dist.destroy_process_group()

# %%
        
def test(opt, task_idx):

    opt.train = False
    device = torch.device('cuda:{}'.format(gpu)) if gpu>=0 else torch.device('cpu')
    model = Pix2PixModel(opt, device)
    model = model.to(device)
    model.eval()
    if task_idx == 3: # if edges2handbags
        opt.direction = 'AtoB'
    test_dataset = AlignedDataset(opt)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,            
            num_workers=4,
            pin_memory=True)

    model.netG = load_pb_conv(model.netG, opt.netG_filter_list, opt.weights, task_idx)

    for i, data in enumerate(test_loader):
        model.set_input(data)
        model.forward()
        model.save_test_images(i)
        print(f"Task {opt.task_num} : Image {i}")
        if i > 50:
            break

    del model
    image_path_list = opt.img_save_path
    image_real = 'real_B'
    image_fake = 'fake_B'
    fid_value = calculate_fid_given_paths(image_path_list, [image_real, image_fake],
                                                            50,
                                                            True,
                                                            2048)
# %%

def main():

    opt = Pix2PixGANOptions().parse()
    tasks = ['cityscapes', 'maps', 'facades', 'edges2handbags']
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    if opt.train:
        
        start_task = opt.st_task_idx
        end_task = len(tasks)

        opt.world_size = len(opt.gpu_ids) * opt.nodes                
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'  

        for task_idx in range(start_task, end_task): 
            # Create Task folder 
            opt.task_folder_name = "Task_"+str(task_idx+1)+"_"+tasks[task_idx]+"_"+"pix2pixGAN"
            opt.image_folder_name = "Intermediate_train_images"
            if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)):
                os.makedirs(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name))

            opt.ckpt_save_path = os.path.join(opt.checkpoints_dir, opt.task_folder_name)
            opt.img_save_path = os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)

            if task_idx == 0:
                netG_filter_list = []
                weights = []
            else:
                old_task_folder_name = "Task_"+str(task_idx)+"_"+tasks[task_idx-1]+"_"+"pix2pixGAN"
                print("Loading ", os.path.join(opt.checkpoints_dir, old_task_folder_name)+'/filters.pt')
                filters = torch.load(os.path.join(opt.checkpoints_dir, old_task_folder_name)+'/filters.pt')
                netG_filter_list = filters["netG_filter_list"]
                weights = filters["weights"]

            opt.netG_filter_list = netG_filter_list
            opt.weights = weights

            opt.dataroot = '../pytorch-CycleGAN-and-pix2pix/datasets/' + tasks[task_idx]
            opt.task_num = task_idx+1
            if tasks[task_idx] == 'edges2handbags': # in case of edges2handbags
                opt.direction = 'AtoB'
            # pdb.set_trace()
            mp.spawn(train, nprocs=len(opt.gpu_ids), args=(opt,))
            if opt.train_continue:
                opt.train_continue=False # to prevent train continue multiple times

    else:
        '''
        We will load the unconstrained filters and the weights ONLY from the last task. 
        This is because, after every task we store the unconstrined filter and weight 
        matrix of that task and all the previous ones. So we will only load from the last one
        which will contain everything we need. 
        '''
        print("In Testing mode")
        start_task = 0
        end_task = len(tasks)
        load_filter_path = opt.checkpoints_dir+f"/Task_{len(tasks)}_{tasks[-1]}_pix2pixGAN/filters.pt"
        opt.load_filter_path = load_filter_path

        filters = torch.load(opt.load_filter_path)
        opt.netG_filter_list = filters["netG_filter_list"]
        opt.weights = filters["weights"]
        opt.image_folder_name = "Test_images"

        for task_idx in range(start_task, end_task):
            print(f"Task {task_idx+1}")

            opt.task_folder_name = "Task_"+str(task_idx+1)+"_"+tasks[task_idx]+"_"+"pix2pixGAN"
            opt.img_save_path = os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)
            if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)):
                    os.makedirs(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name))

            opt.dataroot = '../pytorch-CycleGAN-and-pix2pix/datasets/' + tasks[task_idx]
            opt.task_num = task_idx+1

            test(opt, task_idx)

if __name__ == "__main__":
    main()
