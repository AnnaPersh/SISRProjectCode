import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d
from torch.nn.modules import padding

class DnCNN3(nn.Module):
    def __init__(self, image_channels=3, net_depth=20,bn_momentum=0.4,bn_eps=0.0001):
        super(DnCNN3,self).__init__()
        self.momentum = bn_momentum  #BN layer momentum
        self.eps = bn_eps
        self.image_channels = image_channels
        self.net_depth = net_depth
        #Values we won't adjust for project
        self.n_maps = 64  #Number of feature maps
        self.kernel_size = 3 #Kernel size for convolutional filters
        self.padding = 1 
        self.DnCNN = None
        self.InitializeNetwork()
        
    def InitializeNetwork(self):
        '''
        Initialize the DnCNN based on currently set parameters.
        '''
        net_layers = []
        # input layer
        net_layers.append(nn.Conv2d(in_channels=self.image_channels, out_channels=self.n_maps, kernel_size=self.kernel_size, padding=self.padding, bias=True))
        net_layers.append(nn.ReLU(inplace=True))
        # middle net_layers
        for i in range(self.net_depth-2):
            net_layers.append(nn.Conv2d(in_channels=self.n_maps, out_channels=self.n_maps, kernel_size=self.kernel_size, padding=self.padding, bias=False))
            net_layers.append(nn.BatchNorm2d(self.n_maps, eps=self.eps,momentum=self.momentum))#eps=0.0001
            net_layers.append(ReLU(inplace=True))
        # last layer
        net_layers.append(nn.Conv2d(in_channels=self.n_maps, out_channels=self.image_channels, kernel_size=self.kernel_size,padding=self.padding, bias=False))
        self.DnCNN3 = nn.Sequential(*net_layers)
        self.InitWeights()


    def InitWeights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self,img):
        original_img = img
        out = self.DnCNN3(img)
        return original_img-out
