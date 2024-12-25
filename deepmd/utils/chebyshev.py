from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION

def C_batch(x,num,t):
    """
    it takes a batch of x sample vector, returns the values of fourier base functions (total of num fourier bias) for each xi
    x:
        2D tensor
            shape: (batch, in_dim)
    num: degree+1
        int
    t: type of chebyshev polynomial
        int
    return:
        (3D tensor, 3D tensor)
            shape: (batch, in_dim, num)
    """
    #x: (batch, in_dim, 1)
    x=tf.expand_dims(x,axis=-1)
    #degrees: (1, 1, num)
    degrees=tf.expand_dims(tf.range(0,num,dtype=GLOBAL_TF_PRECISION),[0,1])
    if t==1:
        values=tf.cos(tf.acos(x)*degrees)
    else:
        values=tf.sin(tf.acos(x)*(degrees+1))/tf.sqrt(1-tf.square(x))
    return values

def cheb1_coeff2curve(x_inputs,coeff):
    """
    x_input:
        2D tensor
            shape: (batch, in_dim)
    coeff:
        3D tensor
            shape: (in_dim, out_dim, num)
    return:
        3D tensor
            shape (batch, in_dim, out_dim)
    """
    #num: degree+1, or number of bias
    num=tf.shape(coeff_alpha)[-1]
    #mat: (batch,in_dim,num)
    mat=C_batch(x_inputs,num,1)
    #y: (batch, in_dim, out_dim)
    y=tf.einsum('ijk,jlk->ijl',mat,coeff)
    return y

def cheb2_coeff2curve(x_inputs,coeff):
    """
    x_input:
        2D tensor
            shape: (batch, in_dim)
    coeff_alpha, coeff_beta:
        3D tensor
            shape: (in_dim, out_dim, num)
    return:
        3D tensor
            shape (batch, in_dim, out_dim)
    """
    #num: degree+1, or number of bias
    num=tf.shape(coeff_alpha)[-1]
    #mat: (batch,in_dim,num)
    mat=C_batch(x_inputs,num,2)
    #y: (batch, in_dim, out_dim)
    y=tf.einsum('ijk,jlk->ijl',mat,coeff)
    return y

