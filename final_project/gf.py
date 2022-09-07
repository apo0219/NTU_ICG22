import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
def box_filter( image, r ) :
    # image is a 2D array
    h, w = image.shape
    out = np.zeros_like( image )
    integral = image.cumsum(0).cumsum(1)
    # a b
    # c d
    for i in range( h ) :
        for j in range( w ) :
            left = j - r - 1
            right = j + r if j + r < w else w - 1
            up = i - r - 1
            down = i + r if i + r < h else h - 1
            
            a = integral[up][left] if left >= 0 and up >= 0 else 0
            b = integral[up][right] if up >= 0 else 0
            c = integral[down][left] if left >= 0 else 0
            d = integral[down][right]
            out[i][j] = d - c - b + a
    return out


def guided_filter( I, p, r = 4 , eps = 0.0001, resize = None ) :
    # I : 3*h*w, p : h*w
    h, w = p.shape
    Ir = I[...,0]
    Ig = I[...,1]
    Ib = I[...,2]
    if resize :
        h = h//resize
        w = w//resize
        p = transform.resize( p, ( h, w ), order = 1 )
        Ir = transform.resize( Ir, ( h, w ), order = 1 )
        Ig = transform.resize( Ig, ( h, w ), order = 1 )
        Ib = transform.resize( Ib, ( h, w ), order = 1 )
    A = box_filter( np.ones_like( p ), r )
    mean_I_r = box_filter( Ir, r ) / A
    mean_I_g = box_filter( Ig, r ) / A
    mean_I_b = box_filter( Ib, r ) / A
    mean_p = box_filter( p, r ) / A
    mean_Ip_r = box_filter( Ir * p, r ) / A
    mean_Ip_g = box_filter( Ig * p, r ) / A
    mean_Ip_b = box_filter( Ib * p, r ) / A

    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p
    '''
                rr, rg, rb
    Singma =    rg, gg, gb
                rb, gb, bb 
    '''
    var_I_rr = box_filter( Ir * Ir, r ) / A - mean_I_r * mean_I_r + eps
    var_I_gg = box_filter( Ig * Ig, r ) / A - mean_I_g * mean_I_g + eps
    var_I_bb = box_filter( Ib * Ib, r ) / A - mean_I_b * mean_I_b + eps
    var_I_rg = box_filter( Ir * Ig, r ) / A - mean_I_r * mean_I_g
    var_I_rb = box_filter( Ir * Ib, r ) / A - mean_I_r * mean_I_b
    var_I_gb = box_filter( Ig * Ib, r ) / A - mean_I_g * mean_I_b

    inv_I_rr = var_I_gg * var_I_bb - var_I_gb * var_I_gb
    inv_I_gg = var_I_rr * var_I_bb - var_I_rb * var_I_rb
    inv_I_bb = var_I_rr * var_I_gg - var_I_rg * var_I_rg
    inv_I_rg = var_I_gb * var_I_rb - var_I_rg * var_I_bb
    inv_I_rb = var_I_rg * var_I_gb - var_I_gg * var_I_rb
    inv_I_gb = var_I_rb * var_I_rg - var_I_rr * var_I_gb

    I_cov = inv_I_rr * var_I_rr + inv_I_rg * var_I_rg + inv_I_rb * var_I_rb

    inv_I_rr /= I_cov
    inv_I_gg /= I_cov
    inv_I_bb /= I_cov
    inv_I_rg /= I_cov
    inv_I_rb /= I_cov
    inv_I_gb /= I_cov
    

    ar = inv_I_rr * cov_Ip_r + inv_I_rg * cov_Ip_g + inv_I_rb * cov_Ip_b
    ag = inv_I_rg * cov_Ip_r + inv_I_gg * cov_Ip_g + inv_I_gb * cov_Ip_b
    ab = inv_I_rb * cov_Ip_r + inv_I_gb * cov_Ip_g + inv_I_bb * cov_Ip_b

    b = mean_p - ar * mean_I_r - ag * mean_I_g - ab * mean_I_b

    if resize :
        ar = transform.resize( ar, ( h*resize, w*resize ), order = 1 )
        ag = transform.resize( ag, ( h*resize, w*resize ), order = 1 )
        ab = transform.resize( ab, ( h*resize, w*resize ), order = 1 )
        b = transform.resize( b, ( h*resize, w*resize ), order = 1 )
        A = box_filter( np.ones_like( b ), r )

    out = box_filter( ar, r ) * I[...,0] + box_filter( ab, r ) * I[...,1] + box_filter( ag, r ) * I[...,2] + box_filter( b, r )
    out /= A

    return out
guided = io.imread( "/share/ICG/final/predict/batch1_image0_origin.jpg" )
input = io.imread( "/share/ICG/final/predict/batch1_image0_style1.jpg" )
guided_f = guided.astype( np.float32 ) / 255
input_f = input.astype( np.float32 ) / 255
# guided_tr = guided_f.transpose( 2, 0, 1 )
# input_tr = input_f.transpose( 2, 0, 1 )
# print( guided_tr )
out_r = guided_filter( guided_f, input_f[...,0], r = 10, resize=4 )
print( 'r' )
out_g = guided_filter( guided_f, input_f[...,1], r = 10, resize=4 )
print( 'g' )
out_b = guided_filter( guided_f, input_f[...,2], r = 10, resize=4 )
print( 'b' )
# out_r = guided_filter( input_tr, guided_tr[0,...] )
# print( 'r' )
# out_g = guided_filter( input_tr, guided_tr[1,...] )
# print( 'g' )
# out_b = guided_filter( input_tr, guided_tr[2,...] )
# print( 'b' )
# out_r = ( out_r - np.amin( out_r ) ) / ( np.amax( out_r ) - np.amin( out_r ) )
# out_g = ( out_g - np.amin( out_g ) ) / ( np.amax( out_g ) - np.amin( out_g ) )
# out_b = ( out_b - np.amin( out_b ) ) / ( np.amax( out_b ) - np.amin( out_b ) )
out = np.stack( ( out_r, out_g, out_b ), axis = -1 )
out = np.where( out < 0, 0, out )
out = np.where( out > 1, 1, out )
io.imsave( '/share/ICG/final/fgf_4_r10.jpg', out )
plt.subplot( 1, 3, 1 )
plt.imshow( guided )
plt.subplot( 1, 3, 2 )
plt.imshow( input )
plt.subplot( 1, 3, 3 )
plt.imshow( out )
plt.show()
