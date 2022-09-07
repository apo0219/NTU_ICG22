from myclass import StyleBank, MyLoss, MyConv
import torch
from torch import nn
checkpoint = torch.load( '3l_beta_3e5/epoch37/checkpoint.pth', map_location='cpu' )
model = checkpoint['model']
print( type( model.bank ) )
print( len( model.bank ) )
model.bank.append( 
    nn.Sequential(
        MyConv( 256, 128, 3, 1, 1 ),
        MyConv( 128, 128, 3, 1, 1 ),
        MyConv( 128, 256, 3, 1, 1 )
    )
)
print( type( model.bank ) )
print( len( model.bank ) )
print( type( model ) )
# image = torch.FloatTensor( 5, 3, 512, 512 ).cuda(0)
# style_images = [ torch.FloatTensor( 3, 512, 512 ) for i in range( 5 ) ]
# print( f"image shape : {image.shape}" )
# # create model by denote how many styles
# model = StyleBank( total_style=5 ).cuda(0)
# # for i in range( len( model.bank ) ) :
# #     model.bank[i] = model.bank[i].cuda(0)
# # feed only img into model get autoencoder out
# out_org = model( image )
# print( "out_org shape : ", out_org.shape )
# # feed by style id get out passing style block 
# out_style = model( image, 4 )
# print( "out_style shape : ", out_style.shape )

# # # create loss by denote alpha beta gamma
# # # please to check 5.1 in paper to set the ratio
# # loss = MyLoss( 1, 1, 1 )
# # # feed predict, original image, style image and set mode 
# # loss_origin = loss( out_org, image, None, mode = 'auto' )
# # print( loss_origin.item() )
# # loss_style = loss( out_style, image, style_images[4], mode = 'style' )
# # print( loss_style.item() )
