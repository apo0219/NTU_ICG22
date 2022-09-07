from pathlib import Path
import torch
from math import sqrt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torchvision import models
from copy import deepcopy

class MyImageDataset( Dataset ) :
    def __init__( self, img_list, tar_list, transform=None, change_tar_shape = True ):
        self.img_list = img_list
        self.tar_list = tar_list
        self.transform = transform
        self.totensor = ToTensor()
        self.change_tar_shape = change_tar_shape
    def __len__( self ) :
        return len( self.img_list )
    
    def __getitem__( self, idx ) :
        image = Image.open( self.img_list[idx] )
        tar = Image.open( self.tar_list[idx] )
        if ( self.transform ) :
            image = self.transform( image )
            if ( self.change_tar_shape ) :
                tar = self.transform( tar )
            else :
                tar = self.totensor( tar )
        return image, tar


'''
possible improving
batch normal in twoconv
drop out ( i don't know where to use )
upsampling by convtranspose2d with different par to get 2*size
'''

class MyConv( nn.Module ) :
    def __init__( self, in_channel, out_channel, kernel_size, stride, padding, padding_mode = 'reflect' ) :
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d( in_channel, out_channel, kernel_size = kernel_size, stride = stride, padding = padding, padding_mode = padding_mode ),
            nn.InstanceNorm2d( out_channel ),
            nn.ReLU()
        )
    def forward( self, x ) :
        return self.model( x )

class DenseBlock( nn.Module ) :
    def __init__( self, in_channel, num_layers, growth_rate ):
        super().__init__()
        self.k0 = in_channel
        self.k = growth_rate
        self.num_layers = num_layers
        self.layers = self.make_layer()
        # print( self.layers )
    def make_layer( self ) :
        layer_list = nn.ModuleList([
            nn.Sequential(
                MyConv( self.k0 + i * self.k, self.k * 4, 1, 1, 0 ),
                MyConv( 4 * self.k, self.k, 3, 1, 1 )                        
            ) for i in range( self.num_layers )
        ])
        # layer_list = []
        # for i in range( self.num_layers ) :
        #     layer_list.append(
        #         nn.Sequential(
        #             MyConv( self.k0 + i * self.k, self.k * 4, 1, 1, 0 ),
        #             MyConv( 4 * self.k, self.k, 3, 1, 1 )                        
        #         )
        #     )
        return layer_list
    def forward( self, x ) :
        out = self.layers[0]( x )
        feature = torch.cat( ( x, out ), 1 )
        for i in range( 1, self.num_layers ) :
            out = self.layers[i]( feature )
            feature = torch.cat( ( out, feature ), 1 )
        return feature
class TransitionLayer( nn.Module ) :
    def __init__( self, in_channel, out_channel ):
        super().__init__()
        self.model = nn.Sequential(
            MyConv( in_channel, out_channel, 1, 1, 0 ),
            nn.MaxPool2d( 2 )
        )
    def forward( self, x ) :
        return self.model( x )

class UpConv( nn.Module ) :
    def __init__( self, in_channel, out_channel, num_layers ):
        super().__init__()
        model_list = [
            nn.ConvTranspose2d( in_channel, out_channel, 4, stride = 2, padding = 1 ),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU()
        ]
        for i in range( num_layers ) :
            model_list.append( MyConv( out_channel, out_channel, 3, 1, 1 ) )
        self.model = nn.Sequential( *model_list )
    def forward( self, x ):
        return self.model( x )
class StyleBank( nn.Module ) :
    def __init__( self, total_style ) :
        super().__init__()
        self.encoder = self.Encoder()
        self.decoder = self.Decoder()
        self.bank = nn.ModuleList([
            # nn.Sequential(
            #     DenseBlock( 256, 6, 32 ),
            #     MyConv( 448, 256, 1, 1, 0 )
            # ) for i in range( total_style )
            nn.Sequential(
                MyConv( 256, 128, 3, 1, 1 ),
                MyConv( 128, 128, 3, 1, 1 ),
                MyConv( 128, 256, 3, 1, 1 )
            ) for i in range( total_style )
        ])
    def forward( self, x, style_id = None ):
        out = self.encoder( x )
        if style_id is not None :
            # style_id is a list and len( style_id ) = B
            # should we train same style in the same batch ? ( it's more quick )
            out = self.bank[style_id]( out )
        return self.decoder( out )

    class Encoder( nn.Module ) :
        def __init__( self ) :
            super().__init__()
            self.inlayer = nn.Sequential(
                MyConv( 3, 32, 3, 1, 1 ),
                MyConv( 32, 64, 3, 1, 1 ),
                nn.MaxPool2d( 2 )
            )
            self.d1 = nn.Sequential(
                DenseBlock( 64, 6, 32 ),
                TransitionLayer( 256, 128 )
            )
            self.d2 = nn.Sequential(
                DenseBlock( 128, 6, 32 ),
                TransitionLayer( 320, 256 )
            )
        def forward( self, x ):
            out = self.inlayer( x )
            out = self.d1( out )
            out = self.d2( out )
            return out
    class Decoder( nn.Module ) :
        def __init__( self ) :
            super().__init__()
            self.u1 = UpConv( 256, 128, 6 )
            self.u2 = UpConv( 128, 64, 6 )
            self.outlayer = nn.Sequential(
                nn.ConvTranspose2d( 64, 32, 4, stride = 2, padding = 1 ),
                nn.InstanceNorm2d( 32 ),
                nn.ReLU(),
                MyConv( 32, 32, 3, 1, 1 ),
                MyConv( 32, 32, 3, 1, 1 ),
                nn.Conv2d( 32, 3, kernel_size = 1, stride = 1, padding = 0 ),
                nn.Sigmoid()
            )
        def forward( self, x ) :
            out = self.u1( x )
            out = self.u2( out )
            out = self.outlayer( out )
            return out
class Paper_StyleBank( nn.Module ) :
    def __init__(self, total_style):
        super().__init__()
        self.total_style = total_style

        self.encoder_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(9, 9), stride=2, padding=(4, 4), bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=(9, 9), stride=2, padding=(4, 4), bias=False),
        )

        self.style_bank = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True)
            )
            for i in range(total_style)])
    def forward( self, x, style_id = None ):
        out = self.encoder( x )
        if style_id is not None :
            # style_id is a list and len( style_id ) = B
            # should we train same style in the same batch ? ( it's more quick )
            out = self.bank[style_id]( out )
        return self.decoder( out )

def gram_matrix( x ) :
    b, c, h, w = x.shape
    features = x.view( b * c, h * w )
    G = torch.mm( features, features.t() )
    G = G.div( b * c * h * w )
    return G

class StyleLoss( nn.Module ):
    def __init__( self ) :
        super().__init__()
        self.target = None
        self.mode = 'learn'
        self.mse = nn.MSELoss( reduction = 'mean' )
        self.loss = None
    def forward( self, x ) :
        G = gram_matrix( x )
        if self.mode == 'loss' :
            self.loss = self.weight * self.mse( G, self.target )
        elif self.mode == 'learn' :
            self.target = G.detach()
        return x
    
class ContentLoss( nn.Module ) :
    def __init__( self ) :
        super().__init__()
        self.target = None
        self.mode = 'learn'
        self.mse = nn.MSELoss( reduction = 'mean' )
        self.loss = None
    def forward( self, x ) :
        if self.mode == 'loss' :
            self.loss = self.weight * self.mse( x, self.target )
        elif self.mode == 'learn' :
            self.target = x
        return x

class Normalization( nn.Module ) :
    def __init__( self ) :
        super().__init__()
        # mean = torch.tensor([0.485, 0.456, 0.406])
        # std = torch.tensor([0.229, 0.224, 0.225])
        self.mean = nn.Parameter( torch.tensor( [0.485, 0.456, 0.406] ).view(-1, 1, 1),requires_grad=False )
        self.std = nn.Parameter( torch.tensor( [0.229, 0.224, 0.225] ).view(-1, 1, 1),requires_grad=False )
    def forward( self, img ) :
        return ( img - self.mean ) / self.std

content_layers = ['conv_9']
content_weight = {
	'conv_9': 1
}
style_layers = [ 'conv_2', 'conv_4', 'conv_6', 'conv_9']
style_weight = {
	'conv_2': 1,
	'conv_4': 1,
	'conv_6': 1,
	'conv_9': 1,
}
# vgg16 = models.vgg16( pretrained=True ).features.eval()
vgg16 = models.vgg16()
vgg16.load_state_dict( torch.load( 'vgg16.pth' ) )
vgg16 = vgg16.features.eval()
class LossNet( nn.Module ) :
    def __init__( self ) :
        super().__init__()
        cnn = deepcopy( vgg16 )
        content_losses = []
        style_losses = []
        model = nn.Sequential( Normalization() )
        i = 0
        for layer in cnn.children() :
            if isinstance( layer, nn.Conv2d ) :
                i += 1
                name = f'conv_{i}'
            elif isinstance( layer, nn.ReLU ) :
                name = f'relu_{i}'
                layer = nn.ReLU( inplace = False )
            elif isinstance( layer, nn.MaxPool2d ) :
                name = f'pool_{i}'
            elif isinstance( layer, nn.BatchNorm2d ) :
                name = f'bn_{i}'
            else:
                raise RuntimeError( f"Unrecognized layer : {layer.__class__.__name__}" )
    
            model.add_module( name, layer )
            # print( model )
            if name in content_layers :
                content_loss = ContentLoss()
                content_loss.weight = content_weight[name]
                model.add_module( f'content_loss_{i}', content_loss )
                content_losses.append( content_loss )

            if name in style_layers :
                style_loss = StyleLoss()
                style_loss.weight = style_weight[name]
                model.add_module( f'style_loss_{i}', style_loss )
                style_losses.append( style_loss )
            
        for i in range( len( model ) -1, -1, -1 ):
            if isinstance( model[i], ContentLoss ) or isinstance( model[i], StyleLoss ):
                model = model[:(i + 1)]
                break
        self.model = model
        self.style_losses = style_losses
        self.content_losses = content_losses

    def forward( self, predict, image, style ):

        for layer in self.content_losses :
            layer.mode = 'learn'
        for layer in self.style_losses :
            layer.mode = 'nope'
        self.model( image )

        for layer in self.content_losses :
            layer.mode = 'nope'
        for layer in self.style_losses :
            layer.mode = 'learn'
        self.model( style )

        for layer in self.content_losses :
            layer.mode = 'loss'
        for layer in self.style_losses :
            layer.mode = 'loss'
        self.model( predict )

        content_loss = 0
        style_loss = 0
        for cl in self.content_losses :
            content_loss += cl.loss
        for sl in self.style_losses :
            style_loss += sl.loss
        
        return content_loss, style_loss
class MyLoss( nn.Module ) :
    def __init__( self, alpha, beta, gama ) :
        super().__init__()
        self.mse = nn.MSELoss( reduction = 'mean' )
        self.lossnet = LossNet()
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
    def forward( self, predict, image, style, mode = 'style' ) :
        if mode == 'style' :
            b, c, h, w = predict.shape
            style = torch.stack( [ style for i in range( b ) ] )
            diff_i = torch.sum( torch.abs( predict[...,1:] - predict[...,:-1] ) )
            diff_j = torch.sum( torch.abs( predict[:,:,1:,:] - predict[:,:,:-1,:] ) )
            tv_loss = self.gama * ( diff_i + diff_j )
            content_loss, style_loss = self.lossnet( predict, image, style )
            content_loss = self.alpha * content_loss
            style_loss = self.beta * style_loss
            return content_loss + style_loss + tv_loss
        else :
            return self.mse( predict, image ) 