import argparse
import os 
import sys
import pdb
class Pix2PixGANOptions():
    def __init__(self):
        self.initialized = False
    def initialize(self, parser):
        # folder paths
        parser.add_argument('--checkpoints_dir', type=str, default = "./checkpoints")

        # device settings
        parser.add_argument('--train', action='store_true')
        parser.add_argument('--train_continue', action='store_true')
        parser.add_argument('--nodes', type=int, default=1)
        parser.add_argument('--gpu_ids', type=str, default='4,5', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--nr', type=int, default=0, help='ranking within nodes')
        parser.add_argument('--st_task_idx', type=int, default=0, help='specify start task index')
        
        # model and arch
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--ndf', type=int, default=64)
        parser.add_argument('--netD', type=str, default='basic', help='[basic | n_layers | pixel]')
        parser.add_argument('--netG', type=str, default='unet_256', help='[resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--norm', type=str, default='batch', help='[instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal')
        parser.add_argument('--init_gain', type=float, default=0.02)
        parser.add_argument('--dropout', type=bool, default=False)
        parser.add_argument('--lambda_L1', type=float, default=100.0)

        # train hyper params
        parser.add_argument('--lambda_A', type=float, default=10.0)
        parser.add_argument('--lambda_B', type=float, default=10.0)
        parser.add_argument('--lambda_identity', type=float, default=0.5)
        parser.add_argument('--start_epoch', type=int, default=1)
        parser.add_argument('--n_epochs', type=int, default=100)
        parser.add_argument('--n_epochs_decay', type=int, default=100)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--lr', type=float, default=0.0002)
        parser.add_argument('--lr_policy', type=str, default='linear')
        parser.add_argument('--gan_mode', type=str, default='vanilla', help='[vanilla| lsgan | wgangp]')

        # dataset related options
        parser.add_argument('--pool_size', type=int, default=0)
        parser.add_argument('--direction', type=str, default='BtoA')
        parser.add_argument('--input_nc', type=int, default=3)
        parser.add_argument('--output_nc', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--load_size', type=int, default=286)
        parser.add_argument('--crop_size', type=int, default=256)
        parser.add_argument('--preprocess', type=str, default="resize_and_crop", help='[resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', required=False, type=bool)

        # additional params
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.initialize = True

        parser.add_argument('-taskwise_lambda', action='store_true')
        parser.add_argument('-layerwise_lambda', action='store_true')

        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk only if train phase
        if opt.train:
            expr_dir = opt.checkpoints_dir
            if not os.path.exists(opt.checkpoints_dir):
                os.makedirs(opt.checkpoints_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        self.opt = opt
        return self.opt
    
    # def save_opts(self):
    #     opts_path = os.path.join(self.opt.checkpoints_dir + '/opts.txt')
    #     if not os.path.exists(self.opt.checkpoints_dir):
    #         os.makedirs(self.opt.checkpoints_dir)
    #     with open(opts_path, 'w') as f:
    #         self.print_options()
        # f = open(opts_path, 'w')
        # self.print_options()
        # f.close()

# class CycleGANOptions():
#     def __init__(self):        
        # # folder paths
        # self.checkpoints_dir = "./checkpoints_4tasks"

        # # device settings
        # self.train = True
        # self.train_continue = True
        # self.nodes = 1
        # self.gpu_ids = [0,1,2,3]
        # self.nr = 0 # ranking within nodes

        # # model and arch
        # self.ngf = 64
        # self.ndf = 64
        # self.netD = "basic" # [basic | n_layers | pixel]
        # self.netG = "unet_128" # [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
        # self.norm = "instance" # [instance | batch | none]
        # self.init_type = "normal" # [normal | xavier | kaiming | orthogonal]
        # self.init_gain = 0.02 # scaling factor for normal, xavier and orthogonal.
        # self.dropout = False

        # # train hyperparams
        # self.lambda_A = 10.0
        # self.lambda_B = 10.0
        # self.lambda_identity = 0.5
        # self.start_epoch = 1
        # self.n_epochs = 100
        # self.n_epochs_decay = 100
        # self.beta1 = 0.5
        # self.lr = 0.0002
        # self.lr_policy = "linear"
        # self.gan_mode = "lsgan" # [vanilla| lsgan | wgangp]

        # # dataset related options
        # self.pool_size = 50
        # self.direction = "AtoB"
        # self.input_nc = 3
        # self.output_nc = 3
        # self.batch_size = 2
        # self.load_size = 286
        # self.crop_size = 256
        # self.preprocess = "resize_and_crop" # [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
        # self.no_flip = False

    # def save_opts(self):
    #     opts_path = os.path.join(self.checkpoints_dir + '/opts.txt')
    #     if not os.path.exists(self.checkpoints_dir):
    #         os.makedirs(self.checkpoints_dir)
    #     f = open(opts_path, 'w')
    #     for key, value in self.__dict__.items():
    #         f.write(f"{key}: {value} \n")
    #     f.close()

