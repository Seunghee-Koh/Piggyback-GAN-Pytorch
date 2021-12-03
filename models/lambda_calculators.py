import torch
import torch.nn as nn
from models import networks, cycleGAN
from dataloaders.dataloader import UnalignedDataset

# TODO: Enable two types of lambda : list(different by layer), float(constant for all filters)
# 'lambdas' indicates lambdas in PiggybackConv, and PiggybackTransposeConv.
# Lambdas are applied in convert_piggy_layer function in models.networks.define_G

# TODO: Please copy calculated_lambda before converting as filter_list.
def convert_piggy_layer_with_lambda(module, task_num, filter_list, calculated_lambda):
    module_output = module
    new_filt_list = filter_list
    if isinstance(module, nn.Conv2d) or isinstance(module, networks.PiggybackConv):
        lambdas = calculated_lambda if isinstance(calculated_lambda,
                                                  float) else calculated_lambda.pop(0)
        if task_num == 1:
            unc_filt_list = None
        else:
            unc_filt_list = filter_list[0]
            new_filt_list = filter_list[1:]
	    
        module_output = networks.PiggybackConv(in_channels=module.in_channels,
				      out_channels=module.out_channels,
				      kernel_size=module.kernel_size,
				      stride=module.stride,
				      padding=module.padding,
                                      lambdas=lambdas,
				      task=task_num,
				      unc_filt_list=unc_filt_list
				      )
    elif isinstance(module, nn.ConvTranspose2d) or isinstance(module, networks.PiggybackTransposeConv):
        lambdas = calculated_lambda if isinstance(calculated_lambda,
                                                  float) else calculated_lambda.pop(0)
        if task_num == 1:
            unc_filt_list = None
        else:
            unc_filt_list = filter_list[0]
            new_filt_list = filter_list[1:]
        module_output = networks.PiggybackTransposeConv(in_channels=module.in_channels,
					       out_channels=module.out_channels,
					       kernel_size=module.kernel_size,
					       stride=module.stride,
					       padding=module.padding,
                                               lambdas=lambdas,
					       output_padding=module.output_padding,
					       task=task_num,
					       unc_filt_list=unc_filt_list)
    for name, child in module.named_children():
        module_output.add_module(
                name, convert_piggy_layer_with_lambda(child, task_num,new_filt_list,calculated_lambda))
    del module
    return module_output


# similar to train / test function.
# It doesn't need DDP.
# lambda is calculated before mp.spawn.
# After opt.netG_*_filter_list initialized.
# No need to calculate lambda if there is latest_checkpoint.pt 
def get_task_lambda(opt, gpu, task_idx, max_lambda=1.0):
    assert opt.train_continue == False
    # TODO: opt.train? True? False?
    opt.train=True # To call train dataset 
    device = torch.device('cuda:{}'.format(gpu)) if gpu>=0 else torch.device('cpu')
    model = cycleGAN.CycleGAN(opt, device)
    print(opt.netG_A_filter_list)

    model = model.to(device)
    model.eval()
    # Lambda should be calculated using trainset!
    train_dataset = UnalignedDataset(opt)
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch_size,
            shuffle=False,            
            num_workers=4,
            pin_memory=True)
    cnt = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            model.set_input(data)
            model.forward()
            # Distance measure.. Cycleloss..
            lambda_A = opt.lambda_A
            lambda_B = opt.lambda_B
            loss_cycle_A = model.criterionCycle(model.rec_A, model.real_A) * lambda_A
            loss_cycle_B = model.criterionCycle(model.rec_B, model.real_B) * lambda_B
            loss = loss_cycle_A + loss_cycle_B
            total_loss = total_loss + loss.item()
            cnt = cnt + 1
            if not i%20:
                print(i, total_loss/cnt)

    total_loss = total_loss / cnt
    lambdas = round(total_loss *opt.ngf) * 1.0 / opt.ngf
    return min(lambdas, max_lambda)


        


