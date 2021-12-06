#This script reads training and testing summary JSONs from output directories,
#as well as accesses models as needed for specific plots, to make the plots
#needed for the paper.  Each plot has a set of comments describing what data is
#being loaded and visualized.
#New set of code for each plot starts at a text entry NEW PLOT

import torch
import torch.nn as nn
import DnCNN_train_alp
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
sns.set(font_scale=2.5)
sns.set_style("whitegrid")
sns.axes_style("darkgrid")
xkcd_colors = ['cobalt','light purple','teal'] #'teal','pink','adobe','dark orange']
sns.set_palette(sns.xkcd_palette(xkcd_colors))

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
    
    # NEW PLOT: Plot the training and testing times for scale factor 2 trained and tested BSDS100.
    # Some comments on this:
    #   - The Training time could be reduced per epoch by producing the image patches once
    #      prior to training, rather than processing the images to get patches every epoch.
    #   - The NN output time is the time for an image to go into DnCNN3, and then get the result
    time_eval_dir = "./output_submit/BSDS200SegTraining_AllDatPerEpoch_11242021_run0/"
    training_summary = {}
    testing_summary = {}
    with open("%s/testingsummary_BSDS100Testing.json"%(time_eval_dir),"r") as f:
        testing_summary = json.load(f)
    with open("%s/training_summary_049_epoch"%(time_eval_dir),"r") as f:
        training_summary = json.load(f)
    plt.hist(training_summary["epoch_train_time"])
    plt.title("Distribution of epoch training times \n (100 images from BSDS200, 50x50 patches)")
    plt.xlabel("Epoch training time (sec.)")
    plt.show()

    exec_arr =np.array(testing_summary["epoch_execution_time"])*1000.0 
    plt.hist(exec_arr,range=[np.min(exec_arr),np.max(exec_arr)],bins=20)
    plt.title("Time needed to test image with NN \n (30 images from BSDS100, full images)")
    plt.xlabel("NN output time (ms)")
    plt.show()

    # NEW PLOT: Plot the training and testing times for scale factor 2 trained and tested Monet-Only images.
    # A couple notes on this:
    time_eval_dir = "./output_submit/MonetPaintingTraining_11262021_run0/"
    training_summary = {}
    testing_summary = {}
    with open("%s/testingsummary_MonetTesting.json"%(time_eval_dir),"r") as f:
        testing_summary = json.load(f)
    with open("%s/training_summary_049_epoch"%(time_eval_dir),"r") as f:
        training_summary = json.load(f)
    plt.hist(training_summary["epoch_train_time"])
    plt.title("Distribution of epoch training times \n (20 Monet paintings, 50x50 patches)")
    plt.xlabel("Epoch training time (sec.)")
    plt.show()

    exec_arr =np.array(testing_summary["epoch_execution_time"])*1000.0 
    plt.hist(exec_arr,range=[55,85],bins=20)
    plt.title("Time needed to test image with NN \n (400 x 400 patches from Monet paintings)")
    plt.xlabel("NN output time (ms)")
    plt.show()

    # NEW PLOT: Show an example of epoch 50 original image, blurred image, NN output, and blurred image + NN output
    #           Done here for BSDS200 training and BSDS100 testing.
    val_dataset_dir = 'C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\testing\\BSDS100\\Single\\'
    neural_net_dir = "./output_submit/BSDS200SegTraining_AllDatPerEpoch_11242021_run0/"

    model = DnCNN3()
    model_state = "%smodelstate_049_epoch.json"%(neural_net_dir)
    model.load_state_dict(torch.load(model_state))
    model.eval()
    model = model.to(device)
    scale_factor = 2
    patch_size = 400
    stride_size = 400
    batch_size=1
    val_loader = GetDataLoader(val_dataset_dir,scale_factor,patch_size,stride_size,batch_size,format="whole")
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
        psnr = peak_signal_noise_ratio(orig[0],y[0]-NN_output[0])
        y_shaped = np.moveaxis(y[0],0,2)
        NN_output_shaped = np.moveaxis(NN_output[0],0,2)
        z_shaped= np.moveaxis(orig[0],0,2)
        ssim = structural_similarity(z_shaped,y_shaped-NN_output_shaped,multichannel=True)
        sns.set(font_scale=1.4)
        sns.set_style("white")
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(orig[0].transpose((1,2,0)))
        axarr[0].title.set_text("True original image")
        axarr[1].imshow(y[0].transpose((1,2,0)))
        axarr[1].title.set_text("Down/upsampled image, scale factor %i"%(int(scale_factor)))
        axarr[2].imshow(y[0].transpose((1,2,0)) - NN_output[0].transpose((1,2,0)))
        axarr[2].title.set_text("Down/upsampled image + NN output")
        axarr[2].set_xlabel("epoch 50, PSNR=%s, SSIM=%s"%(str(np.round(psnr,2)),str(np.around(ssim,2))))
        for ax in axarr:
            ax.set_xticks([])
            ax.set_yticks([]) 
        plt.show()
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(v[0].transpose((1,2,0)))
        axarr[0].title.set_text("True feature residual, scale factor %s"%(str(np.around(scale_factor,0))))
        axarr[1].imshow(NN_output[0].transpose((1,2,0)))
        axarr[1].title.set_text("DnCNN3 neural network output, epoch 50")
        for ax in axarr:
            ax.set_xticks([])
            ax.set_yticks([]) 
        plt.show()
        #Same as above, but soomed on bird head
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(orig[0].transpose((1,2,0))[50:150,175:300,:])
        axarr[0].title.set_text("True original image")
        axarr[1].imshow(y[0].transpose((1,2,0))[50:150,175:300,:])
        axarr[1].title.set_text("Down/upsampled image")
        axarr[1].set_xlabel("scale factor %s"%(str(np.round(scale_factor,1))))
        axarr[2].imshow(y[0].transpose((1,2,0))[50:150,175:300,:]- NN_output[0].transpose((1,2,0))[50:150,175:300,:])
        axarr[2].title.set_text("Down/upsampled image + NN output")
        axarr[2].set_xlabel("epoch 50, PSNR=%s, SSIM=%s"%(str(np.round(psnr,2)),str(np.around(ssim,2))))
        for ax in axarr:
            ax.set_xticks([])
            ax.set_yticks([]) 
        plt.show()


    # NEW PLOT: Show an example of epoch 50 original image, blurred image, then outputs from BSD200, Mixed artist, and Monet training
    val_dataset_dir = 'C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\testing\\BSDS100\\Single2\\'
    bsd_dir = "./output_submit/BSDS200SegTraining_AllDatPerEpoch_11242021_run0/"
    monet_dir = "./output_submit/MonetPaintingTraining_11262021_run0/"
    combined_dir = "./output_submit/CombinedPaintingTraining_11252021_run0/"
    model_dict = {"BSD100 Training":"./output_submit/BSDS200SegTraining_AllDatPerEpoch_11242021_run0/", 
                  "Monet Training": "./output_submit/MonetPaintingTraining_11262021_run0/",
                  "Mixed Artist Training":"./output_submit/CombinedPaintingTraining_11252021_run0/"}
    scale_factor = 2
    patch_size = 400
    stride_size = 400
    batch_size=1
    val_loader = GetDataLoader(val_dataset_dir,scale_factor,patch_size,stride_size,batch_size,format="whole")
    have_orig = False
    sns.set(font_scale=0.8)
    sns.set_style("white")
    f, axarr = plt.subplots(1,5)
    for j,model_type in enumerate(model_dict):
        for i, (batch_v, batch_y,orig) in enumerate(val_loader):
            model = DnCNN3()
            model_state = "%smodelstate_049_epoch.json"%(model_dict[model_type])
            model.load_state_dict(torch.load(model_state))
            model.eval()
            model = model.to(device)
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
            psnr = peak_signal_noise_ratio(orig[0],y[0]-NN_output[0])
            y_shaped = np.moveaxis(y[0],0,2)
            NN_output_shaped = np.moveaxis(NN_output[0],0,2)
            z_shaped= np.moveaxis(orig[0],0,2)
            ssim = structural_similarity(z_shaped,y_shaped-NN_output_shaped,multichannel=True)
            if not have_orig:
                axarr[0].imshow(orig[0].transpose((1,2,0))[100:200,50:150,:])
                axarr[0].title.set_text("True original image")
                axarr[1].imshow(y[0].transpose((1,2,0))[100:200,50:150,:])
                axarr[1].title.set_text("Down/upsampled image")
                axarr[1].set_xlabel("scale factor %s"%(str(np.round(scale_factor,0))))
                have_orig = True
            axarr[j+2].imshow(y[0].transpose((1,2,0))[100:200,50:150,:]- NN_output[0].transpose((1,2,0))[100:200,50:150,:])
            axarr[j+2].title.set_text("Down/upsampled image + NN output")
            axarr[j+2].set_xlabel("%s, PSNR=%s, SSIM=%s"%(model_type,str(np.round(psnr,2)),str(np.around(ssim,2))))
    for ax in axarr:
        ax.set_xticks([])
        ax.set_yticks([]) 
    plt.show()


    # NEW PLOT: Show an example of epoch 50 original image, blurred image, NN output, and blurred image + NN output
    #           Done here for Monet training and Monet testing.
    val_dataset_dir = "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\project_data\\Art_benchmark_data\\Monet_Testing\\Single\\"
    neural_net_dir = "./output_submit/MonetPaintingTraining_11262021_run0/"
    model = DnCNN3()
    model_state = "%smodelstate_049_epoch.json"%(neural_net_dir)
    model.load_state_dict(torch.load(model_state))
    model.eval()
    model = model.to(device)
    scale_factor = 2
    patch_size = 400
    stride_size = 400
    batch_size=1
    cropdict = {'xmin':350, 'xmax':600, 'ymin':550, 'ymax':750}
    val_loader = GetDataLoader(val_dataset_dir,scale_factor,patch_size,stride_size,batch_size,format="crop", cropdict = cropdict)
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
        psnr = peak_signal_noise_ratio(orig[0],y[0]-NN_output[0])
        y_shaped = np.moveaxis(y[0],0,2)
        NN_output_shaped = np.moveaxis(NN_output[0],0,2)
        z_shaped= np.moveaxis(orig[0],0,2)
        ssim = structural_similarity(z_shaped,y_shaped-NN_output_shaped,multichannel=True)
        sns.set(font_scale=1.4)
        sns.set_style("white")
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(v[0].transpose((1,2,0)))
        axarr[0].title.set_text("True feature residual patch, scale factor %s"%(str(np.around(scale_factor,1))))
        axarr[1].imshow(NN_output[0].transpose((1,2,0)))
        axarr[1].title.set_text("DnCNN3 neural network output patch, epoch 50")
        for ax in axarr:
            ax.set_xticks([])
            ax.set_yticks([]) 
        plt.show()
        #Same as above, but soomed on bird head
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(orig[0].transpose((1,2,0)))
        axarr[0].title.set_text("True original image patch")
        axarr[1].imshow(y[0].transpose((1,2,0)))
        axarr[1].title.set_text("Down/upsampled image patch \n scale factor %i"%(int(scale_factor)))
        axarr[2].imshow(y[0].transpose((1,2,0))- NN_output[0].transpose((1,2,0)))
        axarr[2].title.set_text("Down/upsampled image patch + NN output")
        axarr[2].set_xlabel("epoch 50, PSNR=%s, SSIM=%s"%(str(np.round(psnr,2)),str(np.around(ssim,2))))
        for ax in axarr:
            ax.set_xticks([])
            ax.set_yticks([]) 
        plt.show()

    # NEW PLOT: Show an example of original image and blurred image, then outputs at epochs 0, 10, 20, 30, 40, and 50 (2x4)
    #           Done here for BSDS200 training and BSDS100 testing sample
    val_dataset_dir = 'C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\testing\\BSDS100\\Single\\'
    neural_net_dir = "./output_submit/BSDS200SegTraining_AllDatPerEpoch_11242021_run0/"
    model_state_list = ["%smodelstate_000_epoch.json"%(neural_net_dir),"%smodelstate_009_epoch.json"%(neural_net_dir),
                        "%smodelstate_019_epoch.json"%(neural_net_dir),"%smodelstate_029_epoch.json"%(neural_net_dir)]
    sns.set(font_scale=1.0)
    sns.set_style("white")
    f, axarr = plt.subplots(1,4)
    for j,model_state in enumerate(model_state_list):
        epoch_num = int(model_state.split("_")[-2])+1
        model = DnCNN3()
        model.load_state_dict(torch.load(model_state))
        model.eval()
        model = model.to(device)
        scale_factor = 2
        val_loader = GetDataLoader(val_dataset_dir,scale_factor,400,400,1,format="whole")
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
            psnr = peak_signal_noise_ratio(orig[0],y[0]-NN_output[0])
            y_shaped = np.moveaxis(y[0],0,2)
            NN_output_shaped = np.moveaxis(NN_output[0],0,2)
            z_shaped= np.moveaxis(orig[0],0,2)
            ssim = structural_similarity(z_shaped,y_shaped-NN_output_shaped,multichannel=True)
            axarr[j].imshow(y[0].transpose((1,2,0)) - NN_output[0].transpose((1,2,0)))
            axarr[j].title.set_text("Down/upsampled image + NN output \n (BSDS200 training)")
            axarr[j].set_xlabel("epoch %i, PSNR=%s, SSIM=%s"%(epoch_num,str(np.round(psnr,2)),str(np.around(ssim,2))))
    for ax in axarr:
        ax.set_xticks([])
        ax.set_yticks([]) 
    plt.show()

     #  NEW PLOT
    #  - Show the PSNR vs. Training Epoch on BSDS100 testing data for scale factor 2, 3, and 4
    #  - Show the SSIM vs. Training Epoch on BSDS100 testing data for scale factor 2,3, and 4
    #  NOTES:
    #  - For each testing result, the neural network was trained on data scaled with the same
    #    scale factor as was applied to the testing data
    sns.set(font_scale=2.5)
    sns.set_style("whitegrid")
    sns.axes_style("darkgrid")
    xkcd_colors = ['cobalt','teal','light purple'] #'teal','pink','adobe','dark orange']
    sns.set_palette(sns.xkcd_palette(xkcd_colors))
    sf2_eval_dir = "./output_submit/BSDS200SegTraining_AllDatPerEpoch_11242021_run0/"
    sf3_eval_dir = "./output_submit/BSDS200SegTraining_AllDatPerEpoch_SF3_1182021_run0/"
    sf4_eval_dir = "./output_submit/BSDS200SegTraining_AllDatPerEpoch_SF4_11272021_run0/"
    sf2_training = {}
    sf2_testing = {}
    sf3_training = {}
    sf3_testing = {}
    sf4_training = {}
    sf4_testing = {}
    with open("%s/testingsummary_BSDS100Testing.json"%(sf2_eval_dir),"r") as f:
        sf2_testing = json.load(f)
    with open("%s/training_summary_049_epoch"%(sf2_eval_dir),"r") as f:
        sf2_training = json.load(f)
    with open("%s/testingsummary_BSDS100Testing.json"%(sf3_eval_dir),"r") as f:
        sf3_testing = json.load(f)
    with open("%s/training_summary_049_epoch"%(sf3_eval_dir),"r") as f:
        sf3_training = json.load(f)
    with open("%s/testingsummary_BSDS100Testing.json"%(sf4_eval_dir),"r") as f:
        sf4_testing = json.load(f)
    with open("%s/training_summary_049_epoch"%(sf4_eval_dir),"r") as f:
        sf4_training = json.load(f)

    plt.plot(sf2_testing["epoch_num"],sf2_testing["epoch_mean_psnr"],linewidth=4,label="Scale factor 2")
    plt.plot(sf3_testing["epoch_num"],sf3_testing["epoch_mean_psnr"],linewidth=4,label="Scale factor 3")
    plt.plot(sf4_testing["epoch_num"],sf4_testing["epoch_mean_psnr"],linewidth=4,label="Scale factor 4")
    plt.legend()
    plt.title("Mean PSNR evaluated on testing data \n (Trained on BSDS200 subset, tested on BSDS100 subset)")
    plt.xlabel("Epoch number in training")
    plt.ylabel("Mean PSNR (db)")
    plt.show()

    plt.plot(sf2_testing["epoch_num"],sf2_testing["epoch_mean_ssim"],linewidth=4,label="Scale factor 2")
    plt.plot(sf3_testing["epoch_num"],sf3_testing["epoch_mean_ssim"],linewidth=4,label="Scale factor 3")
    plt.plot(sf4_testing["epoch_num"],sf4_testing["epoch_mean_ssim"],linewidth=4,label="Scale factor 4")
    plt.legend()
    plt.title("Mean SSIM evaluated on testing data \n (Trained on BSDS200 subset, tested on BSDS100 subset)")
    plt.xlabel("Epoch number in training")
    plt.ylabel("Mean SSIM")
    plt.show()


    # NEW PLOT:  - Show the PSNR vs. training epoch on Monet Only testing for Monet Only training, Mixed training, and BSDS200 training
    #            - Show the SSIM vs. training epoch on Monet Only testing for Monet Only training,Mixed training, and BSDS200 training
    sns.set(font_scale=2.5)
    sns.set_style("whitegrid")
    sns.axes_style("darkgrid")
    xkcd_colors = ['cobalt','teal','light purple'] #'teal','pink','adobe','dark orange']
    sns.set_palette(sns.xkcd_palette(xkcd_colors))
    mo_eval_dir = "./output_submit/MonetPaintingTraining_11262021_run0/"
    mix_eval_dir = "./output_submit/CombinedPaintingTraining_11252021_run0/"
    bsd_eval_dir = "./output_submit/BSDS200SegTraining_AllDatPerEpoch_11242021_run0/"
    mo_training = {}
    mo_testing = {}
    mix_training = {}
    mix_testing = {}
    bsd_training = {}
    bsd_testing = {}
    with open("%s/testingsummary_MonetTesting.json"%(mo_eval_dir),"r") as f:
        mo_testing = json.load(f)
    with open("%s/training_summary_049_epoch"%(mo_eval_dir),"r") as f:
        mo_training = json.load(f)
    with open("%s/testingsummary_MonetTesting.json"%(mix_eval_dir),"r") as f:
        mix_testing = json.load(f)
    with open("%s/training_summary_049_epoch"%(mix_eval_dir),"r") as f:
        mix_training = json.load(f)
    with open("%s/testingsummary_MonetTesting.json"%(bsd_eval_dir),"r") as f:
        bsd_testing = json.load(f)
    with open("%s/training_summary_049_epoch"%(bsd_eval_dir),"r") as f:
        bsd_training = json.load(f)
    plt.plot(mo_testing["epoch_num"],mo_testing["epoch_mean_psnr"],linewidth=4,label="Monet only training")
    plt.plot(mix_testing["epoch_num"],mix_testing["epoch_mean_psnr"],linewidth=4,label="Mixed artist training")
    plt.plot(bsd_testing["epoch_num"],bsd_testing["epoch_mean_psnr"],linewidth=4,label="BSDS200 training")
    plt.legend()
    plt.title("Mean PSNR evaluated on Monet testing data \n (Down/upsample scale factor 2, 50 epochs)")
    plt.xlabel("Epoch number in training")
    plt.ylabel("Mean PSNR (db)")
    plt.show()

    plt.plot(mo_testing["epoch_num"],mo_testing["epoch_mean_ssim"],linewidth=4,label="Monet only training")
    plt.plot(mix_testing["epoch_num"],mix_testing["epoch_mean_ssim"],linewidth=4,label="Mixed artist training")
    plt.plot(bsd_testing["epoch_num"],bsd_testing["epoch_mean_ssim"],linewidth=4,label="BSDS200 training")
    plt.legend()
    plt.title("Mean SSIM evaluated on Monet testing data \n (Down/upsample scale factor 2, 50 epochs)")
    plt.xlabel("Epoch number in training")
    plt.ylabel("Mean SSIM")
    plt.show()

    # NEW PLOT: Load the Caravaggio testing, make plots to evaluate testing on these
    with open("%s/testingsummary_CaravaggioTesting.json"%(mo_eval_dir),"r") as f:
        mo_testing = json.load(f)
    with open("%s/training_summary_049_epoch"%(mo_eval_dir),"r") as f:
        mo_training = json.load(f)
    with open("%s/testingsummary_CaravaggioTesting.json"%(mix_eval_dir),"r") as f:
        mix_testing = json.load(f)
    with open("%s/training_summary_049_epoch"%(mix_eval_dir),"r") as f:
        mix_training = json.load(f)
    with open("%s/testingsummary_CaravaggioTesting.json"%(bsd_eval_dir),"r") as f:
        bsd_testing = json.load(f)
    with open("%s/training_summary_049_epoch"%(bsd_eval_dir),"r") as f:
        bsd_training = json.load(f)
    plt.plot(mo_testing["epoch_num"],mo_testing["epoch_mean_psnr"],linewidth=4,label="Monet only training")
    plt.plot(mix_testing["epoch_num"],mix_testing["epoch_mean_psnr"],linewidth=4,label="Mixed artist training")
    plt.plot(bsd_testing["epoch_num"],bsd_testing["epoch_mean_psnr"],linewidth=4,label="BSDS200 training")
    plt.legend()
    plt.title("Mean PSNR evaluated on Caravaggio testing data \n (Down/upsample scale factor 2, 50 epochs)")
    plt.xlabel("Epoch number in training")
    plt.ylabel("Mean PSNR (db)")
    plt.show()

    plt.plot(mo_testing["epoch_num"],mo_testing["epoch_mean_ssim"],linewidth=4,label="Monet only training")
    plt.plot(mix_testing["epoch_num"],mix_testing["epoch_mean_ssim"],linewidth=4,label="Mixed artist training")
    plt.plot(bsd_testing["epoch_num"],bsd_testing["epoch_mean_ssim"],linewidth=4,label="BSDS200 training")
    plt.legend()
    plt.title("Mean SSIM evaluated on Caravaggio testing data \n (Down/upsample scale factor 2, 50 epochs)")
    plt.xlabel("Epoch number in training")
    plt.ylabel("Mean SSIM")
    plt.show()