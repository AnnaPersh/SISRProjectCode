#lets write a script that makes plots for parameter tuning validation.
#  - load in like 5 images for validating neural networks (use set 14)
#  - load the models with the top 5 PSNRs and top 5 SSIMs
#  - make a plot showing the average PSNR and ssim for these ten models
#     * are the rankings consistent with the training data?

#  - Also make plot of all 75 models showing peak PSNRs and peak SSIMs
#    - for PSNR, have three plots.  2D box plots where bn_momentum on one
#      axis, learn_rate_upbound on the other.  Each plot is for a neural
#      net depth.
#  - Repeat plot with models, but for validation data.

#  - We'll take the model with the best performance on the set 14 data.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import json
import matplotlib.pyplot as plt
import numpy as np
import os,sys
import pandas as pd
import seaborn as sns
import image_preprocessing as imp

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from network_alp import DnCNN3

#Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    val_dataset_dir = "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\testing\\Set14\\"
    pt_dir = ""
    try:
        pt_dir = "./output/%s/"%(sys.argv[1])
    except IndexError:
        print("Give a run name to load parameter tuning NN_outputs from.")
        print("Output directory must have been generated with parameter_tuning_alp.py util.")
        print("usage: parameter_tuning_plots_alp.py [run_name]")
        sys.exit(0)
    if not os.path.isdir(pt_dir):
        print("OUTPUT DIRECTORY DOES NOT EXIST.  EXITING.")
        sys.exit(0)
    pt_summary = {}
    with open("%s/parameter_tuning_summary"%(pt_dir),"r") as f:
        pt_summary=json.load(f)

    pt_models = {}
    for key in pt_summary.keys():
        if type(pt_summary[key]) is not list or key == "training_data_dirs":
            continue
        pt_models[key] = pt_summary[key]
    print(pt_models)
    pt_df = pd.DataFrame(pt_models)
    net_depth_10 = pt_df.loc[pt_df["net_depth"]==10]
    net_depth_15 = pt_df.loc[pt_df["net_depth"]==15]
    net_depth_20 = pt_df.loc[pt_df["net_depth"]==20]

    ssim_net_depth_10 = net_depth_10.pivot("bn_momentum", "learn_rate_upbound", "last_epoch_mean_ssim")
    ssim_net_depth_15 = net_depth_15.pivot("bn_momentum", "learn_rate_upbound", "last_epoch_mean_ssim")
    ssim_net_depth_20 = net_depth_20.pivot("bn_momentum", "learn_rate_upbound", "last_epoch_mean_ssim")
    ax = sns.heatmap(ssim_net_depth_10,annot=True)
    plt.title("Mean SSIM as a function of hyperparameters \n (SSIR Set 5 training, net depth=10, epoch 25)")
    plt.xlabel("Initial learning rate")
    plt.ylabel("Batch normalization momentum")
    plt.show()

    ax = sns.heatmap(ssim_net_depth_15,annot=True)
    plt.title("Mean SSIM as a function of hyperparameters \n (SSIR Set 5 training, net depth=15, epoch 25)")
    plt.xlabel("Initial learning rate")
    plt.ylabel("Batch normalization momentum")
    plt.show()

    ax = sns.heatmap(ssim_net_depth_20,annot=True)
    plt.title("Mean SSIM as a function of hyperparameters \n (SSIR Set 5 training, net depth=20, epoch 25)")
    plt.xlabel("Initial learning rate")
    plt.ylabel("Batch normalization momentum")
    plt.show()

    psnr_net_depth_10 = net_depth_10.pivot("bn_momentum", "learn_rate_upbound", "last_epoch_mean_psnr")
    psnr_net_depth_15 = net_depth_15.pivot("bn_momentum", "learn_rate_upbound", "last_epoch_mean_psnr")
    psnr_net_depth_20 = net_depth_20.pivot("bn_momentum", "learn_rate_upbound", "last_epoch_mean_psnr")
    ax = sns.heatmap(psnr_net_depth_10,annot=True)
    plt.title("Mean PSNR as a function of hyperparameters \n (SSIR Set 5 training, net depth=10, epoch 25)")
    plt.xlabel("Initial learning rate")
    plt.ylabel("Batch normalization momentum")
    plt.show()

    ax = sns.heatmap(psnr_net_depth_15,annot=True)
    plt.title("Mean PSNR as a function of hyperparameters \n (SSIR Set 5 training, net depth=15, epoch 25)")
    plt.xlabel("Initial learning rate")
    plt.ylabel("Batch normalization momentum")
    plt.show()

    ax = sns.heatmap(psnr_net_depth_20,annot=True)
    plt.title("Mean PSNR as a function of hyperparameters \n (SSIR Set 5 training, net depth=20, epoch 25)")
    plt.xlabel("Initial learning rate")
    plt.ylabel("Batch normalization momentum")
    plt.show()

    valdict = {"last_epoch_mean_psnr":[], "last_epoch_mean_ssim":[], "bn_momentum":[], "learn_rate_upbound":[],
               "net_depth":[],"model_id":[]}
    if os.path.exists("%s/parameter_tuning_validation_summary"%(pt_dir)):
        print("LOADING ALREADY EXISTENT VALIDATION DICTIONARY")
        with open("%s/parameter_tuning_validation_summary"%(pt_dir),"r") as f:
            valdict = json.load(f)
    else:
        #Now, we load our validation dataset and evaluate how well each model performs in psnr and ssim
        Preprocessor = imp.ImageDataProcessor(val_dataset_dir)
        Preprocessor.SetScaleFactor(2.0)
        Preprocessor.SetPatchSize(50)
        Preprocessor.SetStrideSize(50)
        #Load patches of images in directory defined for dataloader
        batch_final_patches, proc_patches = Preprocessor.GetAllPNGData(format="patches") #"whole" for full images, "patches" for patches
        #Massage image patches into shape loaded into neural network
        batch_final_patches = Preprocessor.PrepForTorch(batch_final_patches)
        proc_patches = Preprocessor.PrepForTorch(proc_patches)
        SDataset = imp.SISR_Dataset(val_dataset_dir)
        SDataset.SetXElements(batch_final_patches)
        SDataset.SetYElements(proc_patches)
        val_loader = DataLoader(dataset=SDataset, batch_size=1) #drop_last=True, num_workers=1,
        #To do here:
        #  - Load the model for given hyperparameters
        #  - append to bn_momentum, learn_rate_upbound, and net_depth in valdict
        #  - evaluate that model for the whole dataset, append psnr and ssim
        #  - Make same plots as above but for validation
        #  - ???
        #  - Profit
        model = DnCNN3()
        epoch_num = []
        epoch_loss = []
        mean_psnrs = []
        mean_ssims = []
        run_times = []
        criterion = nn.MSELoss(reduction='sum') #mean
        for modid in pt_summary["model_id"]:
            print("TRYING TO LOAD ID: "+str(modid))
            model_name = "modelstate_model_id_%i.json"%(modid)
            model_state = "%s%s"%(pt_dir,model_name)
            valdict["model_id"].append(int(modid))
            pt_model_specs = pt_df.loc[pt_df["model_id"]==modid]
            valdict["bn_momentum"].append(pt_model_specs["bn_momentum"].values[0])
            valdict["learn_rate_upbound"].append(pt_model_specs["learn_rate_upbound"].values[0])
            valdict["net_depth"].append(int(pt_model_specs["net_depth"].values[0]))
            print("trying to load model state: "+str(model_state))
            model = DnCNN3()
            model.net_depth = pt_model_specs["net_depth"].values[0]
            model.InitializeNetwork()
            model.load_state_dict(torch.load(model_state),strict=False)
            model.eval()
            model = model.to(device)
            epoch_psnrs = []
            epoch_ssims = []
            for i, (batch_v, batch_y,batch_x) in enumerate(val_loader):
                batch_y = batch_y.type(torch.FloatTensor)
                batch_v = batch_v.type(torch.FloatTensor)
                batch_x = batch_x.type(torch.FloatTensor)
                batch_y = batch_y.to(device)
                NN_output = model(batch_y) # Should be added onto the the y
                y = batch_y.cpu().detach().numpy() #Downsampled, then upsampled image
                NN_output = NN_output.cpu().detach().numpy() #Downsampled, then upsampled image after going through network
                x = batch_v.cpu().detach().numpy() #True feature residual
                batch_x = batch_x.cpu().detach().numpy()
                for j,entry in enumerate(NN_output):
                    epoch_psnrs.append(peak_signal_noise_ratio(batch_x[j],y[j]-NN_output[j]))
                    y_shaped = np.moveaxis(y[j],0,2)
                    NN_output_shaped = np.moveaxis(NN_output[j],0,2)
                    x_shaped= np.moveaxis(batch_x[j],0,2)
                    epoch_ssims.append(structural_similarity(x_shaped,y_shaped-NN_output_shaped,multichannel=True))
            valdict["last_epoch_mean_psnr"].append(np.mean(epoch_psnrs))
            valdict["last_epoch_mean_ssim"].append(np.mean(epoch_ssims))
        with open("%s/parameter_tuning_validation_summary"%(pt_dir),"w") as f:
            json.dump(valdict,f,indent=4)
    val_df = pd.DataFrame(valdict)
    net_depth_10 = val_df.loc[val_df["net_depth"]==10]
    net_depth_15 = val_df.loc[val_df["net_depth"]==15]
    net_depth_20 = val_df.loc[val_df["net_depth"]==20]

    ssim_net_depth_10 = net_depth_10.pivot("bn_momentum", "learn_rate_upbound", "last_epoch_mean_ssim")
    ssim_net_depth_15 = net_depth_15.pivot("bn_momentum", "learn_rate_upbound", "last_epoch_mean_ssim")
    ssim_net_depth_20 = net_depth_20.pivot("bn_momentum", "learn_rate_upbound", "last_epoch_mean_ssim")
    ax = sns.heatmap(ssim_net_depth_10,annot=True)
    plt.title("Mean SSIM as a function of hyperparameters \n (SSIR Set 14 validation, net depth=10, epoch 25)")
    plt.xlabel("Initial learning rate")
    plt.ylabel("Batch normalization momentum")
    plt.show()

    ax = sns.heatmap(ssim_net_depth_15,annot=True)
    plt.title("Mean SSIM as a function of hyperparameters \n (SSIR Set 14 validation, net depth=15, epoch 25)")
    plt.xlabel("Initial learning rate")
    plt.ylabel("Batch normalization momentum")
    plt.show()

    ax = sns.heatmap(ssim_net_depth_20,annot=True)
    plt.title("Mean SSIM as a function of hyperparameters \n (SSIR Set 14 validation, net depth=20, epoch 25)")
    plt.xlabel("Initial learning rate")
    plt.ylabel("Batch normalization momentum")
    plt.show()

    psnr_net_depth_10 = net_depth_10.pivot("bn_momentum", "learn_rate_upbound", "last_epoch_mean_psnr")
    psnr_net_depth_15 = net_depth_15.pivot("bn_momentum", "learn_rate_upbound", "last_epoch_mean_psnr")
    psnr_net_depth_20 = net_depth_20.pivot("bn_momentum", "learn_rate_upbound", "last_epoch_mean_psnr")
    ax = sns.heatmap(psnr_net_depth_10,annot=True)
    plt.title("Mean PSNR as a function of hyperparameters \n (SSIR Set 14 validation, net depth=10, epoch 25)")
    plt.xlabel("Initial learning rate")
    plt.ylabel("Batch normalization momentum")
    plt.show()

    ax = sns.heatmap(psnr_net_depth_15,annot=True)
    plt.title("Mean PSNR as a function of hyperparameters \n (SSIR Set 14 validation, net depth=15, epoch 25)")
    plt.xlabel("Initial learning rate")
    plt.ylabel("Batch normalization momentum")
    plt.show()

    ax = sns.heatmap(psnr_net_depth_20,annot=True)
    plt.title("Mean PSNR as a function of hyperparameters \n (SSIR Set 14 validation, net depth=20, epoch 25)")
    plt.xlabel("Initial learning rate")
    plt.ylabel("Batch normalization momentum")
    plt.show()
