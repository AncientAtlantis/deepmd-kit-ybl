
from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION

def F_batch(x,num):
    """
    it takes a batch of x sample vector, returns the values of fourier base functions (total of num fourier bias) for each xi
    x:
        2D tensor
            shape: (batch, in_dim)
    num: number of fourier bias, start from 1
        int
    return:
        (3D tensor, 3D tensor)
            shape: (batch, in_dim, num)
    """
    #x: (batch, in_dim, 1)
    x=tf.expand_dims(x,axis=-1)
    #k: (1, 1, num)
    k=tf.expand_dims(tf.range(0,num),axis=0)
    #print(k)
    k=tf.expand_dims(k,axis=0)
    k=tf.cast(k,GLOBAL_TF_FLOAT_PRECISION)
    return tf.cos(k*x),tf.sin(k*x)

def f_coeff2curve(x_inputs,coeff_alpha,coeff_beta):
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
    num=tf.shape(coeff_alpha)[-1]
    #cos, sin: (batch,in_sim,num)
    cos,sin=F_batch(x_inputs,num)
    y=tf.einsum('ijk,jlk->ijl',cos,coeff_alpha)+tf.einsum('ijk,jlk->ijl',sin,coeff_beta)
    #print(y)
    return y

