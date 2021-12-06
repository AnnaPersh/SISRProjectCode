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
import image_preprocessing as imp
from network_alp import DnCNN3

#Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_printoptions(threshold=1000)
class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


if __name__ == '__main__':
    dataset_dirs = ["C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\testing\\Set5"]
    output_dir = ""
    try:
        output_dir = "./output/%s/"%(sys.argv[1])
    except IndexError:
        print("Give a run name.")
        print("usage: parameter_tuning_alp.py [run_name]")
        sys.exit(0)
    if os.path.isdir(output_dir):
        print("WARNING: WRITING TO A DIRECTORY THAT ALREADY EXISTS.  MAY OVERWRITE PAST MODELS")
    else:
        print("MAKING NEW DIRECTORY TO SAVE MODELS AT %s"%(output_dir))
        os.mkdir(output_dir)
    # Hyper Parameters
    mini_batch = 75
    num_epochs_per_datadir = 25  #Num epochs per dataset
    image_channels = 3
    optimizer_gamma = 0.87096 #Factor learning rate is reduced by per epoch
    epoch_modelsave_spacing = 1 # Will save a copy of the model parameters every N epochs
    scale_factor = 2.0
    patch_size = 50
    bn_eps = 0.0001

    learn_rate_upbounds = [0.1,0.01,0.001,0.0001,0.00001]
    bn_momentums = [0.1,0.3,0.5,0.7,0.9]
    net_depths = [10,15,20]

    parameter_training_summary = {"num_epochs":num_epochs_per_datadir*len(dataset_dirs), "last_epoch_loss":[],"train_time":[],
                                  "learn_rate_upbound":[],"batch_size":mini_batch, "bn_momentum":[], "bn_eps":bn_eps, "net_depth":[],
                                  "last_epoch_mean_psnr":[], "last_epoch_mean_ssim":[],"training_data_dirs":dataset_dirs,"gamma":optimizer_gamma,
                                  "scale_factor":scale_factor, "model_id":[]} 
    total_epochs = float(len(dataset_dirs))*num_epochs_per_datadir
    model_num = 0
    for net_depth in net_depths:
        for learn_rate_upbound in learn_rate_upbounds:
            for bn_momentum in bn_momentums:
                parameter_training_summary["net_depth"].append(net_depth)
                parameter_training_summary["learn_rate_upbound"].append(learn_rate_upbound)
                parameter_training_summary["bn_momentum"].append(bn_momentum)
                parameter_training_summary["model_id"].append(model_num)
                print("TRAINING NN WITH NET DEPTH %f, LRU %f, BN MOM %f"%(net_depth, learn_rate_upbound, bn_momentum))
                #Initialize model, optimizer, scheduler, and criterion
                #training_summary = {"epoch_num":[], "epoch_loss":[],"epoch_train_time":[],"initial_lr":learn_rate_upbound,"batch_size":mini_batch,
                #                    "bn_momentum":bn_momentum, "bn_eps":bn_eps, "net_depth":net_depth,"epoch_mean_psnr":[], "epoch_mean_ssim":[],
                #                    "training_data_dirs":dataset_dirs}
                model = DnCNN3(image_channels=image_channels, net_depth=net_depth,bn_momentum=bn_momentum, bn_eps = bn_eps)
                model = model.to(device)
                model.train()
                criterion = nn.MSELoss(reduction='sum') #mean
                #criterion = sum_squared_error() 
                # Loop through datasets, do training
                epoch_num = 0
                for dataset_dir in dataset_dirs:
                    print("DOING %i EPOCHS FOR DATASET %s"%(num_epochs_per_datadir,dataset_dir))
                    Preprocessor = imp.ImageDataProcessor(dataset_dir)
                    Preprocessor.SetScaleFactor(scale_factor)
                    Preprocessor.SetPatchSize(patch_size)
                    if image_channels==1:
                        Preprocessor.color = False
                    #Load patches of images in directory defined for dataloader
                    original_patches, proc_patches = Preprocessor.GetAllPNGData(format="patches")
                    #Massage image patches into shape loaded into neural network
                    original_patches = Preprocessor.PrepForTorch(original_patches)
                    proc_patches = Preprocessor.PrepForTorch(proc_patches)
                    SDataset = imp.SISR_Dataset(dataset_dir)
                    SDataset.SetXElements(original_patches)
                    SDataset.SetYElements(proc_patches)

                    #Run training
                    optimizer = optim.Adam(model.parameters(),lr=learn_rate_upbound) 
                    #optimizer = optim.SGD(model.parameters(),lr=learnrateup,weight_decay=weightdecay,momentum=0.2) #0.1 is the initial learning rate
                    scheduler = ExponentialLR(optimizer, gamma=optimizer_gamma,verbose=False)#last_epoch=numepochs
                    training_loss_arr = []
                    start_time = time.time()
                    for i in range(num_epochs_per_datadir):
                        scheduler.step(epoch_num)
                        print("Epoch [%d / %d]"%(i,total_epochs))
                        train_loader = DataLoader(dataset=SDataset, batch_size=mini_batch,shuffle=True,drop_last=True) #drop_last=True, num_workers=1,
                        epoch_loss = 0
                        epoch_psnrs = []
                        epoch_ssims = []
                        for k, (batch_x, batch_y,batch_z) in enumerate(train_loader):
                            optimizer.zero_grad()    
                            batch_y = batch_y.type(torch.FloatTensor)
                            batch_x = batch_x.type(torch.FloatTensor)
                            batch_z = batch_z.type(torch.FloatTensor)
                            # Should we add zero padding to this? according to paper... use ZeroPad2d or torch.nn.functional.pad(or add pdding mode to Conv2D)
                            batch_y = batch_y.to(device)
                            batch_x = batch_x.to(device)
                            # Forward pass
                            NN_output = model(batch_y)
                            loss = criterion(NN_output, batch_x)  # what if the target is y-x??
                            #loss = criterion(NN_output,batch_y-batch_x)
                            # Backward pass
                            loss.backward()
                            # Optimize
                            optimizer.step()
                            epoch_loss += loss.item()
                            batch_y = batch_y.cpu().detach().numpy() #Downsampled, then upsampled image
                            NN_output = NN_output.cpu().detach().numpy() #Downsampled, then upsampled image after going through network
                            batch_x = batch_x.cpu().detach().numpy() #True feature residual
                            batch_z = batch_z.numpy()
                            if i == num_epochs_per_datadir-1: #last epoch
                                for j,entry in enumerate(NN_output):
                                    epoch_psnrs.append(peak_signal_noise_ratio(batch_z[j],batch_y[j]-NN_output[j]))
                                    y_shaped = np.moveaxis(batch_y[j],0,2)
                                    NN_output_shaped = np.moveaxis(NN_output[j],0,2)
                                    z_shaped= np.moveaxis(batch_z[j],0,2)
                                    epoch_ssims.append(structural_similarity(z_shaped,y_shaped-NN_output_shaped,multichannel=True))
                            if (k + 1) % 5 == 0:
                                print('Step: [% d/% d], Loss: %.4f' % ( k + 1, len(train_loader), loss.item()))
                        if i == num_epochs_per_datadir-1:
                            parameter_training_summary["last_epoch_mean_psnr"].append(np.mean(epoch_psnrs))
                            parameter_training_summary["last_epoch_loss"].append(epoch_loss)
                            parameter_training_summary["last_epoch_mean_ssim"].append(np.mean(epoch_ssims))
                            print("Saving model state and optimizer state for final epoch for model ID %i"%(model_num))
                            output_nnstate_filename = "%s/modelstate_model_id_%i.json" % (output_dir,model_num)
                            output_optimstate_filename = "%s/optimizerstate_model_id_%i.json"%(output_dir,model_num)
                            torch.save(model.state_dict(), output_nnstate_filename)
                            torch.save(optimizer.state_dict(), output_optimstate_filename)
                        epoch_num+=1
                    #Make epoch loss plot
                    parameter_training_summary["train_time"].append(time.time() - start_time)
                model_num+=1
                #Save current copy of parameter tuning summary 
                output_paramtuning_json = "%s/parameter_tuning_summary" % (output_dir)
                with open(output_paramtuning_json,"w") as f:
                    json.dump(parameter_training_summary,f,indent=4)
