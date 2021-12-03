# %%
import torch
import torch.nn as nn
import numpy as np
from configs.cycle_GAN_config import CycleGANOptions
import torch.multiprocessing as mp
import torch.distributed as dist
from dataloaders.dataloader import UnalignedDataset
import torch.utils
import os
from models.cycleGAN import CycleGAN
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
        model = CycleGAN(opt, device)
        model = model.to(device)
 
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        if opt.train_continue:
            state_dict = torch.load(opt.ckpt_save_path+'/latest_checkpoint.pt')  
            model.load_state_dict(state_dict['model'])
            opt.start_epoch = state_dict['epoch']
            opt.train_continue = False # to prevent load check point in another task
        train_dataset = UnalignedDataset(opt)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=rank
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
        print("Length of loader is ",len(train_loader))
        for i, data in enumerate(train_loader):
            model.module.set_input(data)
            model.module.optimize_parameters() 
            if (i+1) % 50 == 0 and gpu<=0:
                print_str = (
                    f"Task: {opt.task_num} | Epoch: {epoch} | Iter: {i+1} | G_A: {model.module.loss_G_A:.5f} | "
                    f"G_B: {model.module.loss_G_B:.5f} | cycle_A: {model.module.loss_cycle_A:.5f} | "
                    f"cycle_B: {model.module.loss_cycle_B:.5f} | idt_A: {model.module.loss_idt_A:.5f} | "
                    f"idt_B: {model.module.loss_idt_B:.5f} | D_A: {model.module.loss_D_A:.5f} | "
                    f"D_B: {model.module.loss_D_A:.5f}" 
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
                   
        make_filter_list(model.module.netG_A, opt.netG_A_filter_list, opt.weights_A, opt.task_num)
        make_filter_list(model.module.netG_B, opt.netG_B_filter_list, opt.weights_B, opt.task_num)
        # print(opt.netG_A_filter_list)

        savedict_task = {'netG_A_filter_list':opt.netG_A_filter_list, 
                            'netG_B_filter_list':opt.netG_B_filter_list,
                            'weights_A':opt.weights_A,
                            'weights_B':opt.weights_B
                        }

        torch.save(savedict_task, opt.ckpt_save_path+'/filters.pt')

        # del netG_A_layer_list
        # del netG_B_layer_list
        del opt.netG_A_filter_list
        del opt.netG_B_filter_list
        del opt.weights_A
        del opt.weights_B
    
    dist.barrier()
    del model

    dist.destroy_process_group()

# %%
        
def test(opt, task_idx):

    opt.train = False
    # device = torch.device('cpu')
    device = torch.device('cuda:{}'.format(gpu)) if gpu>=0 else torch.device('cpu')
    model = CycleGAN(opt, device)
    model = model.to(device) 
    model.eval()
    test_dataset = UnalignedDataset(opt)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,            
            num_workers=4,
            pin_memory=True)

    model.netG_A = load_pb_conv(model.netG_A, opt.netG_A_filter_list, opt.weights_A, task_idx)
    model.netG_B = load_pb_conv(model.netG_B, opt.netG_B_filter_list, opt.weights_B, task_idx)

    for i, data in enumerate(test_loader):
        model.set_input(data)
        model.forward()
        model.save_test_images(i)
        print(f"Task {opt.task_num} : Image {i}")
        if i > 50:
            break

    del model
    image_path_list = opt.img_save_path
    image_real = 'real_A'
    image_fake = 'rec_A'
    fid_value = calculate_fid_given_paths(image_path_list, [image_real, image_fake],
                                                            50,
                                                            True,
                                                            2048)
# %%

def main():

    opt = CycleGANOptions().parse()
    tasks = ['cityscapes', 'maps', 'facades', 'vangogh2photo']
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
            opt.task_folder_name = "Task_"+str(task_idx+1)+"_"+tasks[task_idx]+"_"+"cycleGAN"
            opt.image_folder_name = "Intermediate_train_images"
            if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)):
                os.makedirs(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name))

            opt.ckpt_save_path = os.path.join(opt.checkpoints_dir, opt.task_folder_name)
            opt.img_save_path = os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)

            if task_idx == 0:
                netG_A_filter_list = []
                netG_B_filter_list = []
                weights_A = []
                weights_B = []
            else:
                old_task_folder_name = "Task_"+str(task_idx)+"_"+tasks[task_idx-1]+"_"+"cycleGAN"
                print("Loading ", os.path.join(opt.checkpoints_dir, old_task_folder_name)+'/filters.pt')
                filters = torch.load(os.path.join(opt.checkpoints_dir, old_task_folder_name)+'/filters.pt')
                netG_A_filter_list = filters["netG_A_filter_list"]
                netG_B_filter_list = filters["netG_B_filter_list"]
                weights_A = filters["weights_A"]
                weights_B = filters["weights_B"]

            opt.netG_A_filter_list = netG_A_filter_list
            opt.netG_B_filter_list = netG_B_filter_list
            opt.weights_A = weights_A
            opt.weights_B = weights_B

            opt.dataroot = '../pytorch-CycleGAN-and-pix2pix/datasets/' + tasks[task_idx]
            opt.task_num = task_idx+1  


            opt.task_lambda = 0.25
            if opt.taskwise_lambda:
                if opt.train_continue:
                    state_dict = torch.load(opt.ckpt_save_path + '/latest_checkpoint.pt')
                    opt.task_lambda = state_dict['task_lambda']
                else:
                    from models.lambda_calculators import get_task_lambda
                    # task_lambdas = [0.125, 0.0625, 0.375, 0.5, 0.75, 0.375]
                    # opt.task_lambda = task_lambdas[task_idx]
                    opt.task_lambda = get_task_lambda(opt, 0, task_idx)
                    print(f"Task{task_idx}: lambda {opt.task_lambda}")


            # if task_idx == 1:
            #     from models.networks import define_G
            #     from torchsummary import summary
            #     device='cuda:0'
            #     netG = define_G(3,3,64,'resnet_6blocks','instance',False,'normal',0.02,2,opt.netG_A_filter_list)
            #     netG.to(device)

            #     class Idx():
            #         def __init__(self):
            #             self.idx = 0
            #         def plus(self):
            #             self.idx += 1

            #     def make_filter(network, filters, weights, task_num, conv_idx):
            #         if isinstance(network, PiggybackConv) or isinstance(network, PiggybackTransposeConv):
            #             network.unc_filt.requires_grad = False
            #             if task_num == 1:
            #                 filters.append([network.unc_filt.detach().cpu()])
            #             elif task_num == 2:
            #                 filters[conv_idx.idx].append(network.unc_filt.detach().cpu())
            #                 weights.append([network.weights_mat.detach().cpu()])
            #                 conv_idx.plus()
            #                 #print(f"conv_idx inside function: {conv_idx}")
            #             else:
            #                 filters[task_num-1][conv_idx.idx].append(network.unc_filt.detach().cpu())
            #                 weights[task_num-1][conv_idx.idx].append(network.weights_mat.detach().cpu())
            #                 conv_idx.plus()
            #         print(f"conv_idx inter function: {conv_idx.idx}")

            #         for name, child in network.named_children():
            #             print(f"conv_idx outside function: {conv_idx.idx}")
            #             make_filter(child, filters, weights, task_num, conv_idx)
            #     idx = Idx()
            #     pdb.set_trace()
            #     make_filter_list(netG, opt.netG_A_filter_list, opt.weights_A, opt.task_num)
            #     pdb.set_trace()
            mp.spawn(train, nprocs=len(opt.gpu_ids), args=(opt,))

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
        load_filter_path = opt.checkpoints_dir+f"/Task_{len(tasks)}_{tasks[-1]}_cycleGAN/filters.pt"
        opt.load_filter_path = load_filter_path

        filters = torch.load(opt.load_filter_path)
        opt.netG_A_filter_list = filters["netG_A_filter_list"]
        opt.netG_B_filter_list = filters["netG_B_filter_list"]
        opt.weights_A = filters["weights_A"]
        opt.weights_B = filters["weights_B"]
        opt.image_folder_name = "Test_images"

        for task_idx in range(start_task, end_task):
            print(f"Task {task_idx+1}")

            opt.task_folder_name = "Task_"+str(task_idx+1)+"_"+tasks[task_idx]+"_"+"cycleGAN"
            opt.img_save_path = os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)
            if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)):
                    os.makedirs(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name))

            opt.dataroot = '../pytorch-CycleGAN-and-pix2pix/datasets/' + tasks[task_idx]
            opt.task_num = task_idx+1
            test(opt, task_idx)

if __name__ == "__main__":
    main()
