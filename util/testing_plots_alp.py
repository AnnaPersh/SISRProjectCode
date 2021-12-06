import torch
import torch.nn as nn
import DnCNN_train_alp
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    #val_dataset_dir = 'C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\SISR dataset\\testing\\BSDS100\\'
    #val_dataset_dir = "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\project_painting_data\\Monet_Testing\\"
    val_dataset_dir = "C:\\Users\\yinli\\Desktop\\Project_DLTP\\data\\project_painting_data\\Caravaggio_Testing\\"
    #neural_net_dir = "./output/FullSet5Training_OptimizedParameters_11232021/"
    #neural_net_dir = "./output/CombinedPaintingTraining_11252021_run0/"
    #neural_net_dir = "./output/MonetPaintingTraining_11262021_run0/"
    neural_net_dir = "./output/BSDS200SegTraining_AllDatPerEpoch_11242021_run0/"
    output_testingsummary_json = "%s/testingsummary.json"%(neural_net_dir)
    patch_size = 400 
    scale_factor = 2.0
    #C:\Users\yinli\Desktop\Project_DLTP\code\DnCNN DLTP\output
    testing_summary = {"epoch_num":[], "epoch_loss":[],"epoch_execution_time":[],
                        "patch_size": patch_size, "epoch_mean_psnr":[], "epoch_mean_ssim":[],
                        "testing_data_dir":val_dataset_dir}
    training_summary = {}
    with open("%s/training_summary_049_epoch"%(neural_net_dir),"r") as f:
        training_summary = json.load(f)
    if not os.path.exists(output_testingsummary_json):
        print("NO TESTING SUMMARY EXISTS. EVALUATING")
        # Get our validation dataset
        Preprocessor = imp.ImageDataProcessor(val_dataset_dir)
        Preprocessor.SetScaleFactor(scale_factor) #MAKE SURE THIS IS THE SAME AS THE TRAINING
        Preprocessor.SetPatchSize(patch_size)
        Preprocessor.SetStrideSize(patch_size)
        #Load patches of images in directory defined for dataloader
        original_patches, proc_patches = Preprocessor.GetAllPNGJPEGData(format="patches")
        #Massage image patches into shape loaded into neural network
        original_patches = Preprocessor.PrepForTorch(original_patches)
        proc_patches = Preprocessor.PrepForTorch(proc_patches)
        SDataset = imp.SISR_Dataset(val_dataset_dir)
        SDataset.SetXElements(original_patches)
        SDataset.SetYElements(proc_patches)
        val_loader = DataLoader(dataset=SDataset, batch_size=1) #drop_last=True, num_workers=1,

        #Now, loop through all epoch model states and see how the PSNR, SSIM, epoch_loss look 
        print("Accessing all models output during training...")
        model = DnCNN3()
        model_state_list = glob.glob("%smodelstate_*"%(neural_net_dir))
        print("MODEL STATE LIST: %s"%(str(model_state_list)))
        epoch_num = []
        epoch_loss = []
        mean_psnrs = []
        mean_ssims = []
        run_times = []
        criterion = nn.MSELoss(reduction='sum') #mean

        for h,model_state in enumerate(model_state_list):
            model.load_state_dict(torch.load(model_state))
            model.eval()
            model = model.to(device)
            epoch = int(model_state.split("_")[-2])
            print("MODEL STATE FILENAME: %s"%(model_state))
            print("LOOKING AT EPOCH: %i"%(epoch))
            epoch_num.append(epoch)
            epoch_psnrs = []
            epoch_ssims = []
            loss = 0
            for i, (batch_v, batch_y,orig) in enumerate(val_loader):
                batch_y = batch_y.type(torch.FloatTensor)
                batch_v = batch_v.type(torch.FloatTensor)
                orig = orig.type(torch.FloatTensor)
                batch_y = batch_y.to(device)
                start_time = time.time()
                NN_output = model(batch_y)
                NN_output = NN_output.cpu() 
                loss += criterion(NN_output,batch_v).item()
                testing_summary["epoch_execution_time"].append(time.time() - start_time)
                y = batch_y.cpu().detach().numpy() #Downsampled, then upsampled image
                NN_output = NN_output.detach().numpy() #Downsampled, then upsampled image after going through network
                v = batch_v.cpu().detach().numpy() #True feature residual
                orig = orig.cpu().detach().numpy()
                for j,entry in enumerate(NN_output):
                    epoch_psnrs.append(peak_signal_noise_ratio(orig[j],y[j]-NN_output[j]))
                    y_shaped = np.moveaxis(y[j],0,2)
                    NN_output_shaped = np.moveaxis(NN_output[j],0,2)
                    z_shaped= np.moveaxis(orig[j],0,2)
                    epoch_ssims.append(structural_similarity(z_shaped,y_shaped-NN_output_shaped,multichannel=True))
                if i == 0 and h % 10 == 0:
                    f, axarr = plt.subplots(1,3)
                    axarr[0].imshow(y[0].transpose((1,2,0)))
                    axarr[0].title.set_text("Downsampled then upsampled image")
                    axarr[1].imshow(y[0].transpose((1,2,0)) - NN_output[0].transpose((1,2,0)))
                    axarr[1].title.set_text("Downsampled-upsampled image \n plus NN output (epoch %i)"%(h))
                    axarr[2].imshow(orig[0].transpose((1,2,0)))
                    axarr[2].title.set_text("True original image")
                    plt.show()
                    f, axarr = plt.subplots(1,2)
                    axarr[0].imshow(v[0].transpose((1,2,0)))
                    axarr[0].title.set_text("True feature residual")
                    axarr[1].imshow(NN_output[0].transpose((1,2,0)))
                    axarr[1].title.set_text("DnCNN3 neural network output \n epoch %i"%(h))
                    plt.show()
            mean_psnrs.append(np.mean(epoch_psnrs))
            mean_ssims.append(np.mean(epoch_ssims))
            epoch_loss.append(loss)

        testing_summary["epoch_num"] = epoch_num
        testing_summary["epoch_loss"] = epoch_loss
        testing_summary["epoch_mean_psnr"] = mean_psnrs
        testing_summary["epoch_mean_ssim"] = mean_ssims
        with open(output_testingsummary_json,"w") as f:
            json.dump(testing_summary,f,indent=4)
    else:
        print("LOADING EXISTING TESTING SUMMARY")
        with open(output_testingsummary_json,"r") as f:
            testing_summary = json.load(f)
    plt.plot(testing_summary["epoch_num"],testing_summary["epoch_loss"],marker='o',label="testing data")
    plt.plot(training_summary["epoch_num"],training_summary["epoch_loss"],marker='o',label="training data")
    plt.legend()
    plt.title("Total criterion loss for each epoch of model")
    plt.xlabel("Epoch number in training")
    plt.ylabel("Epoch loss")
    plt.show()

    plt.hist(training_summary["epoch_train_time"])
    plt.title("Time needed to train an epoch")
    plt.xlabel("Epoch training time (sec.)")
    plt.show()

    plt.hist(testing_summary["epoch_execution_time"])
    plt.title("Time needed to process image with DnCNN3 \n Image size %i x %i"%(patch_size,patch_size))
    plt.xlabel("Epoch training time (sec.)")
    plt.show()

    plt.plot(training_summary["epoch_num"],training_summary["epoch_mean_psnr"],marker='o',label="training data")
    plt.plot(testing_summary["epoch_num"],testing_summary["epoch_mean_psnr"],marker='o',label="testing data")
    plt.legend()
    plt.title("Mean PSNR evaluated on \n training and testing data")
    plt.xlabel("Epoch number in training")
    plt.ylabel("Mean PSNR (db)")
    plt.show()

    plt.plot(training_summary["epoch_num"],training_summary["epoch_mean_ssim"],marker='o',label="training data")
    plt.plot(testing_summary["epoch_num"],testing_summary["epoch_mean_ssim"],marker='o',label="testing data")
    plt.legend()
    plt.title("Mean SSIM evaluated on \n training and testing data")
    plt.xlabel("Epoch number in training")
    plt.ylabel("Mean SSIM")
    plt.show()

