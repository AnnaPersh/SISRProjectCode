from __future__ import print_function, division
#torch imports
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms, utils
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR,ExponentialLR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from collections import deque

# other standard imports
import os,sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

#local imports
import lib.image_preprocessing as imp
from lib.network_alp import DnCNN3

#Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # constants
    #dataset_dirs = ['C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\testing\\Set5']
    dataset_dirs = ["C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\training\\BSDS200_Segmented\\Part5",
                    "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\training\\BSDS200_Segmented\\Part6",
                    "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\training\\BSDS200_Segmented\\Part3",
                    "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\training\\BSDS200_Segmented\\Part4",
                    "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\training\\BSDS200_Segmented\\Part1",
                    "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\training\\BSDS200_Segmented\\Part2",
                    "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\training\\BSDS200_Segmented\\Part7"]
    #dataset_dirs = ["C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\project_painting_data\\Monet_Training\\Part1",
    #                "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\project_painting_data\\Monet_Training\\Part2",
    #                "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\project_painting_data\\Monet_Training\\Part3",
    #                "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\project_painting_data\\Monet_Training\\Part4",
    #                "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\project_painting_data\\Monet_Training\\Part5"]
    dataset_dirs = deque(dataset_dirs)
    output_dir = ""
    try:
        output_dir = "./output/%s/"%(sys.argv[1])
    except IndexError:
        print("Give a run name.")
        print("usage: dncnn_train_alp.py [run_name]")
        sys.exit(0)
    if os.path.isdir(output_dir):
        print("WARNING: WRITING TO A DIRECTORY THAT ALREADY EXISTS.  MAY OVERWRITE PAST MODELS")
    else:
        print("MAKING NEW DIRECTORY TO SAVE MODELS AT %s"%(output_dir))
        os.mkdir(output_dir)

    # Hyper Parameters
    learn_rate_upbound = 0.001 #paper starts at 0.1, parameter tuning suggests 0.001 near best
    bn_momentum = 0.9 #parameter tuning suggests 0.9 near best
    net_depth = 20 #paper uses 20, parameter tuning on training suggests 10; staying with 20 

    mini_batch = 75
    num_epochs = 50  #Num epochs per dataset
    image_channels = 3
    bn_eps = 0.0001
    optimizer_gamma = 0.87096 #Factor learning rate is reduced by per epoch
    epoch_modelsave_spacing = 1 # Will save a copy of the model parameters every N epochs
    scale_factor = 3.0
    patch_size = 50
    drop_patch_factor = 1  #Reduce the number of patches actually loaded/evaluated by this factor
                           #Tuning this can reduce the number of patches seen per epoch, as well
                           #as help reduce the training time 

    #Initialize model, optimizer, scheduler, and criterion
    training_summary = {"epoch_num":[], "epoch_loss":[],"epoch_train_time":[],"initial_lr":learn_rate_upbound,"batch_size":mini_batch,
                        "bn_momentum":bn_momentum, "bn_eps":bn_eps, "net_depth":net_depth,"epoch_mean_psnr":[], "epoch_mean_ssim":[],
                        "training_data_dirs":dataset_dirs}
    model = DnCNN3(image_channels=image_channels, net_depth=net_depth,bn_momentum=bn_momentum, bn_eps = bn_eps)
    model = model.to(device)
    model.train()
    criterion = nn.MSELoss(reduction='sum') #mean
    training_summary["learnrate_type"]="exponential"

    # Loop through datasets, do training
    #Run training
    optimizer = optim.Adam(model.parameters(),lr=learn_rate_upbound) 
    scheduler = ExponentialLR(optimizer, gamma=optimizer_gamma,verbose=False)#last_epoch=numepochs
    training_loss_arr = []
    for epoch_num in range(num_epochs):
        epoch_loss = 0
        epoch_psnrs = []
        epoch_ssims = []
        start_time = time.time()
        scheduler.step(epoch_num) 
        training_summary["epoch_num"].append(epoch_num)
        for dataset_dir in dataset_dirs:
            print("LOADING DATASET %s NOW FOR EPOCH %i"%(dataset_dir,epoch_num))
            Preprocessor = imp.ImageDataProcessor(dataset_dir)
            Preprocessor.SetScaleFactor(scale_factor)
            Preprocessor.SetPatchSize(patch_size)
            if image_channels==1:
                Preprocessor.color = False
            #Load patches of images in directory defined for dataloader
            original_patches, proc_patches = Preprocessor.GetAllPNGJPEGData(format="patches")
            #Massage image patches into shape loaded into neural network
            original_patches = Preprocessor.PrepForTorch(original_patches)
            proc_patches = Preprocessor.PrepForTorch(proc_patches)
            SDataset = imp.SISR_Dataset(dataset_dir)
            SDataset.SetXElements(original_patches)
            SDataset.SetYElements(proc_patches)
            train_loader = DataLoader(dataset=SDataset, batch_size=mini_batch,shuffle=True,drop_last=True) #drop_last=True, num_workers=1,
    
            print("Epoch [%d / %d]"%(epoch_num,num_epochs))
            for k, (batch_v, batch_y,batch_x) in enumerate(train_loader):
                if k % drop_patch_factor == 0:
                    optimizer.zero_grad()    
                    batch_y = batch_y.type(torch.FloatTensor)
                    batch_v = batch_v.type(torch.FloatTensor)
                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_y = batch_y.to(device)
                    batch_v = batch_v.to(device)
                    # Forward pass
                    NN_output = model(batch_y)
                    loss = criterion(NN_output, batch_v) 
                    # Backward pass
                    loss.backward()
                    # Optimize
                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_y = batch_y.cpu().detach().numpy() #Downsampled, then upsampled image
                    NN_output = NN_output.cpu().detach().numpy() #Neural network output (estimated feature residual)
                    batch_v = batch_v.cpu().detach().numpy() #True feature residual
                    batch_x = batch_x.numpy()
                    for j,entry in enumerate(NN_output):
                        epoch_psnrs.append(peak_signal_noise_ratio(batch_x[j],batch_y[j]-NN_output[j]))
                        y_shaped = np.moveaxis(batch_y[j],0,2)
                        NN_output_shaped = np.moveaxis(NN_output[j],0,2)
                        x_shaped= np.moveaxis(batch_x[j],0,2)
                        epoch_ssims.append(structural_similarity(x_shaped,y_shaped-NN_output_shaped,multichannel=True))
                    if (k + 1) % 5 == 0:
                        print('Step: [% d/% d], Loss: %.4f' % ( k + 1, len(train_loader), loss.item()))
        if epoch_num % epoch_modelsave_spacing == 0:
            print("Saving model state and optimizer state for epoch %i"%(epoch_num))
            output_nnstate_filename = "%s/modelstate_%03d_epoch.json" % (output_dir,epoch_num)
            output_optimstate_filename = "%s/optimizerstate_%03d_epoch.json"%(output_dir,epoch_num)
            torch.save(model.state_dict(), output_nnstate_filename)
            torch.save(optimizer.state_dict(), output_optimstate_filename)
        training_summary["epoch_loss"].append(epoch_loss)
        print("EPOCH'S TOTAL LOSS: "+str(epoch_loss))
        epoch_train_time = time.time() - start_time
        print("EPOCH FINISHED TRAINING AFTER %f SEC"%(epoch_train_time))
        training_summary["epoch_train_time"].append(epoch_train_time)
        training_summary["epoch_mean_psnr"].append(np.mean(epoch_psnrs))
        training_summary["epoch_mean_ssim"].append(np.mean(epoch_ssims))
        dataset_dirs.rotate(1) #Mix up what dataset the NN sees first in the next epoch 
    
    #Make some summary plots to see if the training went well
    plt.plot(training_summary["epoch_num"],training_summary["epoch_loss"],marker='o')
    plt.xlabel("Epoch number")
    plt.ylabel("Training loss")
    plt.show()

    plt.hist(training_summary["epoch_train_time"])
    plt.ylabel("Time needed to train an epoch (sec.)")
    plt.show()

    plt.plot(training_summary["epoch_num"],training_summary["epoch_mean_psnr"],marker='o')
    plt.xlabel("Epoch number")
    plt.ylabel("Mean PSNR (db)")
    plt.show()

    plt.plot(training_summary["epoch_num"],training_summary["epoch_mean_ssim"],marker='o')
    plt.xlabel("Epoch number")
    plt.ylabel("Mean SSIM")
    plt.show()

    #Save summary and model from training
    output_nn_filename = "%s/model_%03d.pth" % (output_dir,epoch_num)
    output_nnstate_filename = "%s/modelstate_%03d_epoch" % (output_dir,epoch_num)
    output_optimstate_filename = "%s/optimizerstate_%03d_epoch"%(output_dir,epoch_num)
    output_trainingsummary_json = "%s/training_summary_%03d_epoch" % (output_dir,epoch_num)
    torch.save(model.state_dict(), output_nnstate_filename)
    torch.save(optimizer.state_dict(), output_optimstate_filename)
    torch.save(model, output_nn_filename)
    torch.save(optimizer.state_dict(), output_optimstate_filename)
    training_summary["training_data_dirs"] = list(dataset_dirs)
    with open(output_trainingsummary_json,"w") as f:
        json.dump(training_summary,f,indent=4)
