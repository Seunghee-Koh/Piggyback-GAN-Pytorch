import torch
import torch.nn as nn
from models import networks, cycleGAN, pix2pix_model
from dataloaders.dataloader import UnalignedDataset, AlignedDataset
from torch.nn.functional import l1_loss

# TODO: Enable two types of lambda : list(different by layer), float(constant for all filters)
# 'lambdas' indicates lambdas in PiggybackConv, and PiggybackTransposeConv.
# Lambdas are applied in convert_piggy_layer function in models.networks.define_G


def consume_prefix_in_state_dict_if_present(state_dict, prefix):
    """Strip the prefix in state_dict in place, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)



# TODO: Please copy calculated_lambda before converting as filter_list.
def convert_piggy_layer_layerwise_lambda(module, task_num, filter_list, calculated_lambda):
    module_output = module
    if isinstance(module, nn.Conv2d) or isinstance(module, networks.PiggybackConv):
        lambdas = calculated_lambda if isinstance(calculated_lambda,
                                                  float) else calculated_lambda.pop(0)
        if task_num == 1:
            unc_filt_list = None
        else:
            unc_filt_list = filter_list.pop(0)
	    
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
            unc_filt_list = filter_list.pop(0)
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
                name, convert_piggy_layer_layerwise_lambda(child, task_num, filter_list,calculated_lambda))
    del module
    return module_output


# similar to train / test function.
# It doesn't need DDP.
# lambda is calculated before mp.spawn.
# After opt.netG_*_filter_list initialized.
# No need to calculate lambda if there is latest_checkpoint.pt 
def get_task_lambda(opt, opt_task_lambda, gpu, max_lambda=1.0):
    assert opt.train_continue == False
    # TODO: opt.train? True? False?
    opt.train=True # To call train dataset 
    device = torch.device('cuda:{}'.format(gpu)) if gpu>=0 else torch.device('cpu')
    model = pix2pix_model.Pix2PixModel(opt_task_lambda, device)
    # model.netG = networks.load_pb_conv(model.netG, opt_task_lambda.netG_filter_list, opt_task_lambda.weights, opt_task_lambda.task_num-1)
    tasks = ['cityscapes', 'maps', 'facades']
    ckpt_path = opt.checkpoints_dir+f"/Task_{opt_task_lambda.task_num}_{tasks[opt_task_lambda.task_num-1]}_pix2pixGAN/latest_checkpoint.pt"
    print(f"Load {ckpt_path} to infer the taskwise lambda.")
    state_dict = torch.load(ckpt_path)
    consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.')
    model.load_state_dict(state_dict['model'])
    model = model.to(device)

    model.eval()
    # Lambda should be calculated using trainset!
    train_dataset = AlignedDataset(opt)
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
            loss = l1_loss(model.fake_B, model.real_B)/2
            total_loss = total_loss + loss.item()
            cnt = cnt + 1
            # if not i%20:
            #     print(i, total_loss/cnt)

    avg_loss = total_loss / cnt
    lambdas = round(avg_loss *opt.ngf) * 1.0 / opt.ngf
    return min(lambdas, max_lambda)

