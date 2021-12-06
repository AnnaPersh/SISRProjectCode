#This script reads training and testing summary JSONs from output directories,
#as well as accesses models as needed for specific plots, to make the plots
#needed for the paper.  Each plot has a set of comments describing what data is
#being loaded and visualized.

import torch
import torch.nn as nn
import DnCNN_train_alp
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
sns.set(font_scale=2.5)
sns.set_style("whitegrid")
sns.axes_style("darkgrid")
#xkcd_colors = ['teal','dark teal','light purple','purple','adobe','red']
xkcd_colors = ['cobalt','light purple','teal'] #'teal','pink','adobe','dark orange']
sns.set_palette(sns.xkcd_palette(xkcd_colors))


from skimage.metrics import peak_signal_noise_ratio, structural_similarity
#
import os
import json
import time
import numpy as np
import glob
#local imports
import lib.image_preprocessing as imp
from lib.network_alp import DnCNN3

#Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def GetDataLoader(datadir,scale_factor,patch_size,stride_size,batch_size,format="whole",cropdict=None):
    Preprocessor = imp.ImageDataProcessor(datadir)
    Preprocessor.SetPatchSize(patch_size)
    Preprocessor.SetStrideSize(patch_size)
    Preprocessor.SetScaleFactor(scale_factor) #MAKE SURE THIS IS THE SAME AS THE TRAINING
    #Load patches of images in directory defined for dataloader
    original_patches, proc_patches = Preprocessor.GetAllPNGJPEGData(format=format,cropdict=cropdict)
    #Massage image patches into shape loaded into neural network
    original_patches = Preprocessor.PrepForTorch(original_patches)
    proc_patches = Preprocessor.PrepForTorch(proc_patches)
    SDataset = imp.SISR_Dataset(datadir)
    SDataset.SetXElements(original_patches)
    SDataset.SetYElements(proc_patches)
    val_loader = DataLoader(dataset=SDataset, batch_size=batch_size) #drop_last=True, num_workers=1,
    return val_loader

if __name__ == "__main__":

    #I forgot to make the training/NN execution times on set 5 and set 14.  Let's make those real quick
    #
    pt_dir = "./output/parameter_tuning_run0/"
    training_summary = {}
    testing_summary = {}
    with open("%s/parameter_tuning_summary"%(pt_dir),"r") as f:
        training_summary = json.load(f)
    training_times = training_summary["train_time"]
    nd10_times = np.array(training_summary["train_time"])[0:25]/25.0
    nd15_times = np.array(training_summary["train_time"])[25:50]/25.0
    nd20_times = np.array(training_summary["train_time"])[50:75]/25.0
    xkcd_colors = ['cobalt'] 
    sns.set_palette(sns.xkcd_palette(xkcd_colors))
    plt.title("Average training time per epoch, net depth 10 \n (Set 5 data, 50x50 patches, stride 10)")
    plt.hist(nd10_times, range=[9.95,10.12], bins = 25)
    plt.xlabel("Epoch training time (sec.)")
    plt.show()
    xkcd_colors = ['light purple'] 
    sns.set_palette(sns.xkcd_palette(xkcd_colors))
    plt.title("Average training time per epoch, net depth 15 \n (Set 5 data, 50x50 patches, stride 10)")
    plt.hist(nd15_times, range=[15.2,15.45], bins = 25)
    plt.xlabel("Epoch training time (sec.)")
    plt.show()
    xkcd_colors = ['teal'] 
    sns.set_palette(sns.xkcd_palette(xkcd_colors))
    plt.title("Average training time per epoch, net depth 20 \n (Set 5 data, 50x50 patches, stride 10)")
    plt.hist(nd20_times, range=[20.45,20.75], bins = 25)
    plt.xlabel("Epoch training time (sec.)")
    plt.show()


    #Now, let's load up a net depth 10, 15, and 20 model and process it on 50x50 patches, just do stride 10
    dataset_dir = 'C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\testing\\Set14'
    scale_factor = 2
    patch_size = 250
    stride_size = 10
    batch_size=1
    val_loader = GetDataLoader(dataset_dir,scale_factor,patch_size,stride_size,batch_size,format="patches")
    model = DnCNN3()
    model.net_depth = 10
    model.InitializeNetwork()
    model_state = "%smodelstate_model_id_15.json"%(pt_dir)
    model.load_state_dict(torch.load(model_state))
    model.eval()
    model = model.to(device)
    test_times = []
    xkcd_colors = ['cobalt'] 
    sns.set_palette(sns.xkcd_palette(xkcd_colors))
    for j in range(0,10):
        for i, (batch_v, batch_y,orig) in enumerate(val_loader):
            batch_y = batch_y.type(torch.FloatTensor)
            batch_v = batch_v.type(torch.FloatTensor)
            orig = orig.type(torch.FloatTensor)
            batch_y = batch_y.to(device)
            start_time = time.time()
            NN_output = model(batch_y)
            NN_output = NN_output.cpu() 
            y = batch_y.cpu().detach().numpy() #Downsampled, then upsampled image
            NN_output = NN_output.detach().numpy() #Downsampled, then upsampled image after going through network
            v = batch_v.cpu().detach().numpy() #True feature residual
            orig = orig.cpu().detach().numpy()
            test_times.append(time.time() - start_time)
    test_times = np.array(test_times)*1000.0
    plt.hist(test_times,range=[0,20],bins=25)
    plt.xlabel("NN execution time (ms)")
    plt.title("Time to test data on NN, net depth 10 \n (Set 14 data, 250x250 patches)")
    plt.show()

    model = DnCNN3()
    model.net_depth = 15
    model.InitializeNetwork()
    model_state = "%smodelstate_model_id_40.json"%(pt_dir)
    model.load_state_dict(torch.load(model_state))
    model.eval()
    model = model.to(device)
    scale_factor = 2
    patch_size = 250
    stride_size = 10
    batch_size=1
    test_times = []
    xkcd_colors = ['light purple'] 
    sns.set_palette(sns.xkcd_palette(xkcd_colors))
    for j in range(0,10):
        for i, (batch_v, batch_y,orig) in enumerate(val_loader):
            batch_y = batch_y.type(torch.FloatTensor)
            batch_v = batch_v.type(torch.FloatTensor)
            orig = orig.type(torch.FloatTensor)
            batch_y = batch_y.to(device)
            start_time = time.time()
            NN_output = model(batch_y)
            NN_output = NN_output.cpu() 
            y = batch_y.cpu().detach().numpy() #Downsampled, then upsampled image
            NN_output = NN_output.detach().numpy() #Downsampled, then upsampled image after going through network
            v = batch_v.cpu().detach().numpy() #True feature residual
            orig = orig.cpu().detach().numpy()
            test_times.append(time.time() - start_time)
    test_times = np.array(test_times)*1000.0
    plt.hist(test_times,range=[0,33],bins=35)
    plt.xlabel("NN execution time (ms)")
    plt.title("Time to test data on NN, net depth 15 \n (Set 14 data, 250x250 patches)")
    plt.show()

    model = DnCNN3()
    model.net_depth = 20
    model.InitializeNetwork()
    model_state = "%smodelstate_model_id_65.json"%(pt_dir)
    model.load_state_dict(torch.load(model_state))
    model.eval()
    model = model.to(device)
    scale_factor = 2
    patch_size = 250
    stride_size = 10
    batch_size=1
    test_times = []
    xkcd_colors = ['teal'] 
    sns.set_palette(sns.xkcd_palette(xkcd_colors))
    for j in range(0,10):
        for i, (batch_v, batch_y,orig) in enumerate(val_loader):
            batch_y = batch_y.type(torch.FloatTensor)
            batch_v = batch_v.type(torch.FloatTensor)
            orig = orig.type(torch.FloatTensor)
            batch_y = batch_y.to(device)
            start_time = time.time()
            NN_output = model(batch_y)
            NN_output = NN_output.cpu() 
            y = batch_y.cpu().detach().numpy() #Downsampled, then upsampled image
            NN_output = NN_output.detach().numpy() #Downsampled, then upsampled image after going through network
            v = batch_v.cpu().detach().numpy() #True feature residual
            orig = orig.cpu().detach().numpy()
            test_times.append(time.time() - start_time)
    test_times = np.array(test_times)*1000.0
    plt.hist(test_times,range=[0,60],bins=25)
    plt.xlabel("NN execution time (ms)")
    plt.title("Time to test data on NN, net depth 20 \n (Set 14 data, 250x250 patches)")
    plt.show()