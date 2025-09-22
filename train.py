import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from nets import get_model_from_name
from torch.utils.data import DataLoader
from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import download_weights, get_classes, get_lr_scheduler, set_optimizer_lr, show_config, weights_init
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda            = True
    distributed     = False
    sync_bn         = False
    fp16            = False
    classes_path    = 'model_data/cls_classes.txt' 
    input_shape     = [224, 224]

    #------------------------------------------------------#
    #   mobilenetv2、
    #   resnet18、resnet34、resnet50、resnet101、resnet152
    #   vgg11、vgg13、vgg16、vgg11_bn、vgg13_bn、vgg16_bn、
    #   vit_b_16、
    #   swin_transformer_tiny、swin_transformer_small、swin_transformer_base
    #------------------------------------------------------#
    backbone        = "ITSShipFormer"
    pretrained      = False
    model_path      = ""


    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 16
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 16
    Freeze_Train        = True
    
    Init_lr             = 5e-4
    Min_lr              = Init_lr * 0.001

    #------------------------------------------------------------------#
    #   optimizer_type  adam、sgd
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 5e-4

    #------------------------------------------------------------------#
    #   lr_decay_type   step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"

    #------------------------------------------------------------------#
    #   save_period     save weight every save_period epochs
    #------------------------------------------------------------------#
    save_period         = 20
    save_dir            = 'logs'
    num_workers         = 4

    train_annotation_path   = "cls_train.txt"
    test_annotation_path    = 'cls_test.txt'

    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0


    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)


    class_names, num_classes = get_classes(classes_path)



    if backbone == "AlexNet" or "GoogLeNet"  or "resnet50":
        model = get_model_from_name[backbone](num_classes=num_classes)

    elif backbone != "vit"  or "ITSShipFormer":
        model = get_model_from_name[backbone](num_classes = num_classes, pretrained = pretrained)
    else:
        model = get_model_from_name[backbone](input_shape = input_shape, num_classes = num_classes, pretrained = pretrained)

    if not pretrained:
        weights_init(model)
    if model_path != "":
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        

        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None
        

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:

            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
        
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(test_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)
    
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

    wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
    total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1


    UnFreeze_flag = False

    if Freeze_Train:
        model.freeze_backbone()


    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size


    nbs             = 64
    lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
    lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
    if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
        nbs             = 256
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-5 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    
    optimizer = {
        'adam'  : optim.Adam(model_train.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay=weight_decay),
        'sgd'   : optim.SGD(model_train.parameters(), Init_lr_fit, momentum = momentum, nesterov=True)
    }[optimizer_type]
    

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
    

    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Error")

    train_dataset   = DataGenerator(train_lines, input_shape, True)
    val_dataset     = DataGenerator(val_lines, input_shape, False)
    
    if distributed:
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        batch_size      = batch_size // ngpus_per_node
        shuffle         = False
    else:
        train_sampler   = None
        val_sampler     = None
        shuffle         = True
        
    gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, 
                            drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
    gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                            drop_last=True, collate_fn=detection_collate, sampler=val_sampler)
    
    #---------------------------------------#
    for epoch in range(Init_Epoch, UnFreeze_Epoch):

        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            nbs             = 64
            lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
            if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
                nbs             = 256
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-5 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
            
            model.Unfreeze_backbone()

            epoch_step      = num_train // batch_size
            epoch_step_val  = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("Error")

            if distributed:
                batch_size = batch_size // ngpus_per_node

            gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
            gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=detection_collate, sampler=val_sampler)

            UnFreeze_flag = True

        if distributed:
            train_sampler.set_epoch(epoch)
            
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)

    if local_rank == 0:
        loss_history.writer.close()
