from __future__ import print_function, division
import os
from scipy._lib.doccer import doc_replace
import torch
import glob
import pandas as pd
from skimage import io, transform,color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# ignore warning
import warnings
warnings.filterwarnings("ignore")

class ImageDataProcessor():
    '''
    Class for accessing all image data in a set directory.  Can return either full images or patches of
    images given a specificed patch size.  Can also convert lists of patches into tensors ready for loading
    to a Dataset class.
    '''
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.img_list = self.GetRootPNGJPEGFileList()
        self.scale_factor = 2.0
        self.patch_size = 50
        self.stride_size = 10 # from the paper
        self.color = True

    def GetRootPNGJPEGFileList(self):
        '''
        Returns a list of the absolute path to all PNG files
        in a given directory.  
        Inputs:
            directory [string]
            File path to a directory containing PNG files
        '''
        png_file_list = glob.glob(self.root_dir+"\\*.png")
        jpeg_file_list = glob.glob(self.root_dir+"\\*.jpg")
        file_list = png_file_list + jpeg_file_list
        return file_list

    def _GetRootImages(self):
        '''
        Returns a list of image objects with the same indexing as
        that of the image file list.
        Outputs:
            Images [list]
            List of images (type np.ndarray) from the currently data directory.
        '''
        Images = []
        for img in self.img_list:
            if self.color:
                Images.append(io.imread(img)/255.)
            else:
                Images.append(color.rgb2gray(io.imread(img)/255.))
        return np.array(Images)
    
    def SetScaleFactor(self,scalefac):
        '''
        Sets scale factor for upscaling and downscaling of images in processing.
        '''
        self.scale_factor = scalefac

    def SetPatchSize(self, patchsize):
        self.patch_size = patchsize

    def SetStrideSize(self,stridesize):
        self.stride_size = stridesize

    def DsUsImage(self,img):
        '''
        Process an image with a simple downscale then upscale.
        Upscale will return image to the original size.
        Note that the order variable has the following semantics: 
        0: Nearest-neighbor 1: Bi-linear (default) 2: Bi-quadratic 
        3: Bi-cubic 4: Bi-quartic 5: Bi-quintic.
        Inputs:
            image [ndarray]
            RGB or grayscale image to downscale then upscale.
        Outputs:
            proc_image [ndarray]
            downscaled and upscaled version of original image.
        '''
        shp = img.shape
        ds_image = []
        proc_image = []
        if self.color:
            ds_image = transform.resize(img,(int(shp[0]/self.scale_factor), int(shp[1]/self.scale_factor), shp[2]),anti_aliasing=True,order=3)
            proc_image = transform.resize(ds_image, (shp[0], shp[1], shp[2]))
        else:
            ds_image = transform.resize(img,(int(shp[0]/self.scale_factor), int(shp[1]/self.scale_factor)),anti_aliasing=True,order=3)
            proc_image = transform.resize(ds_image, (shp[0], shp[1]))
        return proc_image

    def GetPatches(self,img):
        '''
        Given a single image, return an array of images which are patches of the image.
        inputs:
            img [np.array]
            numpy array of dimension 3 (x_size, y_size,num_channels)
        output:
            patches [list of np.arrays]
            patches of dimension 3 (patch_size, patch_size, num_channels)
        '''
        img_shape = img.shape
        patches = []
        for x in np.arange(0,img_shape[0]-self.patch_size+1,self.stride_size):
            for y in np.arange(0,img_shape[1]-self.patch_size+1,self.stride_size):
                if self.color:
                    patch = np.expand_dims(img,axis=2)
                patch=img[x:x+self.patch_size,y:y+self.patch_size,:]
                if self.color and patch.shape == (self.patch_size,self.patch_size,3):
                    patches.append(img[x:x+self.patch_size,y:y+self.patch_size,:])
                if self.color and patch.shape != (self.patch_size,self.patch_size,3):
                    print("Had a weird edge condition.  shape is %s"%(str(patch.shape)))
        #  in the data generator code there is a discard code, need to check that out
        return patches

    def GetAllPNGJPEGData(self,format="patches",cropdict=None):
        '''
        Get all PNG images in the directory set for the image preprocessor.
        Also gets downscaled, then upscaled, versions of the images.

        Inputs:
            format [string]
            Format specifies whether to return a list of patches of images
            (format="patches") or a list of whole images (format="whole").

            cropdict [dict]
            If format="crop", only this patch will be extracted from each image
            loaded.  A dictionary must be provided with the following keys:
            xmin,xmax,ymin,ymax.  Values should be integers.
            WARNING: No error handling yet if range is outside image size bounds.
        Outputs:
            all_patches
        '''
        all_orig_data = []
        all_dsus_data = []
        directory_images = self._GetRootImages()
        directory_dsus_images = []
        for image in directory_images:
            directory_dsus_images.append(self.DsUsImage(image))

        if format is "whole":
            all_orig_data = directory_images
            all_dsus_data = directory_dsus_images
        if format is "crop":
            for j,image in enumerate(directory_images):
                all_orig_data.append(image[cropdict['xmin']:cropdict['xmax'],cropdict['ymin']:cropdict['ymax'],:])
            for j,image in enumerate(directory_dsus_images):
                all_dsus_data.append(image[cropdict['xmin']:cropdict['xmax'],cropdict['ymin']:cropdict['ymax'],:])
        if format is "patches":
            for j,image in enumerate(directory_images):
                orig_patches = self.GetPatches(image)
                dsus_patches = self.GetPatches(directory_dsus_images[j])
                all_orig_data = all_orig_data + orig_patches 
                all_dsus_data = all_dsus_data + dsus_patches
        all_orig_data = np.array(all_orig_data)
        all_dsus_data = np.array(all_dsus_data)
        return all_orig_data, all_dsus_data
    
    def PrepForTorch(self,img_array):
        '''
        takes a numpy array of images and prepares them for use in a pytorch neural network.
        '''
        img_array = np.stack(img_array,axis=0)
        img_array = torch.from_numpy(img_array.transpose((0, 3, 1, 2)))
        return img_array

class SISR_Dataset(Dataset):
    """SISR dataset.""" 
    def __init__(self,root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with dataset stats.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(SISR_Dataset, self).__init__()
        self.root_dir = root_dir
        self.x_elements = [] #nparray of shape (numpatches, patch_size,patch_size,3)
        self.y_elements = []

    def __len__(self):
        return len(self.x_elements)
        
# work in progress
    def __getitem__(self,idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist() 
        x = self.x_elements[idx]
        y = self.y_elements[idx]
        v = y - x
        return v,y,x

    def TrimElements(self,xmin,xmax,ymin,ymax):
        '''
        Trims all elements in the x and y arrays to a pre-defined range.
        If any image can not have a region trimmed in the given ranges, the
        function aborts and leaves x and y elements alone.
        '''
        new_x = []
        new_y = []
        for j,elem in enumerate(self.x_elements):
            new_x.append(elem[xmin:xmax, ymin:ymax, :])
        for j,elem in enumerate(self.y_elements):
            new_y.append(elem[xmin:xmax, ymin:ymax, :])
        self.x_elements = new_x
        self.y_elements = new_y

    def SetXElements(self,xelements):
        '''
        Sets the original images used to train the neural network.
        Inputs:
            xelements [list of np.arrays]
            List of numpy arrays with images that the neural network learns to make.
        '''
        self.x_elements = xelements 

    def SetYElements(self,yelements):
        '''
        Sets the downscaled, then upscaled images used to train the neural network.
        Inputs:
            xelements [list of np.arrays]
            List of numpy arrays with images that the neural network learns to convert
            to the respective original y element.
        '''
        self.y_elements = yelements
