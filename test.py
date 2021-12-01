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
from models.networks import PiggybackConv, PiggybackTransposeConv, load_pb_conv
import copy 
import sys
from pytorch_model_summary import summary as summary_
from models import networks 
import itertools
from utils.utils import ImageBuffer
from utils.utils import save_image, tensor2im
from torchvision.utils import make_grid
import argparse
import pdb
from models.networks import define_G
from models.cycleGAN import CycleGAN
from configs.cycle_GAN_config import CycleGANOptions
from utils.fid_score import calculate_fid_given_paths

def main():
#     start_task = 0
    opt = CycleGANOptions().parse()
    # tasks = ['cityscapes', 'maps', 'facades']
    # torch.manual_seed(0)
    # np.random.seed(0)
    # torch.cuda.manual_seed(0)
    # torch.cuda.manual_seed_all(0)

    # if opt.train:
        
    #     start_task = 0
    #     end_task = len(tasks)

    #     opt.world_size = len(opt.gpu_ids) * opt.nodes                
    #     os.environ['MASTER_ADDR'] = 'localhost'              
    #     os.environ['MASTER_PORT'] = '8888'  

    #     for task_idx in range(start_task, end_task): 
            
    #         # Create Task folder 
    #         opt.task_folder_name = "Task_"+str(task_idx+1)+"_"+tasks[task_idx]+"_"+"cycleGAN"
    #         opt.image_folder_name = "Intermediate_train_images"
    #         if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)):
    #             os.makedirs(os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name))

    #         opt.ckpt_save_path = os.path.join(opt.checkpoints_dir, opt.task_folder_name)
    #         opt.img_save_path = os.path.join(opt.checkpoints_dir, opt.task_folder_name, opt.image_folder_name)

    #         if task_idx == 0:
    #             netG_A_filter_list = []
    #             netG_B_filter_list = []
    #             weights_A = []
    #             weights_B = []
    #         else:
    #             old_task_folder_name = "Task_"+str(task_idx)+"_"+tasks[task_idx-1]+"_"+"cycleGAN"
    #             print("Loading ", os.path.join(opt.checkpoints_dir, old_task_folder_name)+'/filters.pt')
    #             filters = torch.load(os.path.join(opt.checkpoints_dir, old_task_folder_name)+'/filters.pt')
    #             netG_A_filter_list = filters["netG_A_filter_list"]
    #             netG_B_filter_list = filters["netG_B_filter_list"]
    #             weights_A = filters["weights_A"]
    #             weights_B = filters["weights_B"]

            
    #         opt.netG_A_filter_list = netG_A_filter_list
    #         opt.netG_B_filter_list = netG_B_filter_list
    #         opt.weights_A = weights_A
    #         opt.weights_B = weights_B

    #         opt.dataroot = '../pytorch-CycleGAN-and-pix2pix/datasets/' + tasks[task_idx]
    #         opt.task_num = task_idx+1   

    #         device = 'cuda:0'
    #         model = CycleGAN(opt, device).to(device)

    #         pdb.set_trace()
    # real_A = torch.ones(1, 3,64,64)
    # fake_B = real_A*0.67
    # rec_A = real_A*0.33
    # idt_B = real_A*0
    # real_B = -torch.ones(1, 3,64,64)
    # fake_A = real_B*0.67
    # rec_B = real_B*0.33
    # idt_A = real_B*0
    # grid = make_grid(
    #         torch.cat([real_A, fake_B, rec_A, idt_B, 
    #                     real_B, fake_A, rec_B, idt_A], dim=0))
    # save_image(tensor2im(grid.unsqueeze(0)), f"./test.png")

    # image_path_list = './checkpoints/Task_1_cityscapes_cycleGAN/Test_images/'
    # image_real = 'real_A'
    # image_fake = 'rec_A'
    # fid_value = calculate_fid_given_paths(image_path_list, [image_real, image_fake],
    #                                                         50,
    #                                                         True,
    #                                                         2048)
    # print(f'fid value is {fid_value}')

if __name__ == "__main__":
    main()
