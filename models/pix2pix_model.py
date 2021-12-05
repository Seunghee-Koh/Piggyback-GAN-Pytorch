# %%
import torch
import torch.nn as nn
from models import networks 
import itertools
from utils.utils import ImageBuffer
from utils.utils import save_image, tensor2im
from torchvision.utils import make_grid

# %%
class Pix2PixModel(nn.Module):
    def __init__(self, opt, device):
        super(Pix2PixModel, self).__init__()

        self.device = device
        self.opt = opt

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.train:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      opt.dropout, opt.init_type, opt.init_gain, opt.task_num, opt.netG_filter_list, opt.task_lmabda)

        if self.train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(self.opt.input_nc + opt.output_nc, self.opt.ndf, self.opt.netD, 
                                            self.opt.norm, self.opt.init_type, self.opt.init_gain)

        if self.train:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_train_images(self, epoch):
        # save_image(tensor2im(self.real_A), self.opt.img_save_path + f"/real_A_epoch_{epoch}.png")
        # save_image(tensor2im(self.real_B), self.opt.img_save_path + f"/real_B_epoch_{epoch}.png")
        # save_image(tensor2im(self.rec_A), self.opt.img_save_path + f"/rec_A_epoch_{epoch}.png")
        # save_image(tensor2im(self.rec_B), self.opt.img_save_path + f"/rec_B_epoch_{epoch}.png")
        # save_image(tensor2im(self.idt_A), self.opt.img_save_path + f"/idt_A_epoch_{epoch}.png")
        # save_image(tensor2im(self.idt_B), self.opt.img_save_path + f"/idt_B_epoch_{epoch}.png")

        # save image gird
        grid = make_grid(
            torch.cat([self.real_A, self.fake_B, self.real_B], dim=0),
                        nrow=self.opt.batch_size)
        save_image(tensor2im(grid.unsqueeze(0)), self.opt.img_save_path + f"/output_gird_epoch_{epoch:04d}.png")

    def save_test_images(self, idx):
        save_image(tensor2im(self.real_A), self.opt.img_save_path + f"/img_{idx:04d}_real_A.png")
        save_image(tensor2im(self.fake_B), self.opt.img_save_path + f"/img_{idx:04d}_trans_A2B.png")
        save_image(tensor2im(self.real_B), self.opt.img_save_path + f"/img_{idx:04d}_real_B.png")

        # save image gird
        grid = make_grid(
            torch.cat([self.real_A, self.fake_B, self.real_B], dim=0),
                        nrow=1)
        save_image(tensor2im(grid.unsqueeze(0)), self.opt.img_save_path + f"/output_gird_epoch_{idx:04d}.png")
    
    def update_learning_rate(self):
        
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
