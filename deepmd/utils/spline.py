
from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION

#import tensorflow.compat.v1 as tf
#GLOBAL_TF_FLOAT_PRECISION=tf.float64

def B_batch(x,grids,k):
    """
    it takes a batch of x sample vector, returns the values of b-spline base functions (num=G+k) of order k defined at grids for each x points
    x:
        2D tensor
            shape: (num of spline or num of batch, num of x sample points that is equal to in_dim)
    grids:
        2D tensor
            shape: (in_dim, G+2k+1)
    k:
        int
        order of b-spline function, start from 0
    return:
        3D tensor
            shape: (num of spline, num of x sample points/in_dim, G+k)

    """
    #x: (batch, in_dim, 0)
    x=tf.expand_dims(x,axis=2)
    #grids: (0, in_dims, G+2k+1)
    grids=tf.expand_dims(grids,axis=0)
    if k==0:
        values=tf.cast((x>=grids[:,:,:-1]) & (x<grids[:,:,1:]),GLOBAL_TF_FLOAT_PRECISION)
    else:
        B_kml=B_batch(x[:,:,0],grids[0],k-1)
        values=B_kml[:,:,:-1]*(x - grids[:,:,:-(k + 1)])/(grids[:,:,k:-1]-grids[:,:,:-(k+1)])+\
               B_kml[:,:,1:]*(grids[:,:,k+1:]-x)/(grids[:,:,k+1:]-grids[:,:,1:-k])

    return values


def curve2coeff(x_inputs,y_out,grids,k,lamb=1e-8):
    """
    it tackes x_inputs and target y_out, return the fitted coeff of base functions
    the x_inputs specify the x samples for each spline function and each input dimension of the layer

    x_input:
        2D tensor
            shape: (the number of x points for a single spline function, in_dim for the spline matrix of one layer)
    y_out:
        3D tensor
            shape: (the number of x points for a single spline function,in_dim,out_dim)
    grids:
        2D tensor
            shape: (in_dim, G+2k+1)
    k:
        int

    return:
        3D tensor
            shape (in_dim, out_dim, G+k)
    """
    
    out_dim=tf.shape(y_out)[2]

    mat=B_batch(x_inputs,grids,k)
    mat=tf.transpose(mat,perm=[1,0,2])
    mat=tf.expand_dims(mat,axis=1)
    mat=tf.tile(mat,[1,out_dim,1,1])

    y_out=tf.transpose(y_out,perm=[1, 2, 0])
    y_out=tf.expand_dims(y_out,axis=3)
    #print(y_out.shape)
    #print(tf.transpose(mat,perm=[0,1,3,2]).shape)

    XtX=tf.einsum('ijmn,ijnp->ijmp',tf.transpose(mat,perm=[0,1,3,2]),mat)
    Xty=tf.einsum('ijmn,ijnp->ijmp',tf.transpose(mat,perm=[0,1,3,2]),y_out)

    n1=tf.shape(XtX)[0]
    n2=tf.shape(XtX)[1]
    n=tf.shape(XtX)[2]

    identity=tf.eye(n,n,batch_shape=[n1,n2],dtype=GLOBAL_TF_FLOAT_PRECISION)
    A=XtX+lamb*identity
    B=Xty

    A_pinv=tf.linalg.pinv(A)
    coef=tf.matmul(A_pinv, B)[..., 0]
    return coef

def coeff2curve(x_inputs,grids,k,coeff):
    """
    it tackes x_inputs and target y_out, return the fitted coeff of base functions
    the x_inputs specify the x samples for each spline function and each input dimension of the layer

    x_input:
        2D tensor
            shape: (number of batch, in_dim)
    grids:
        2D tensor
            shape: (in_dim, G+2k+1)
    k:
        int
    coeff:
        3D tensor
            shape: (in_dim, out_dim, G+k)
    return:
        3D tensor
            shape (batch or number of x points for a single spline function, in_dim, out_dim)
    """
    #spline: (batch,in_dim,G+k)
    #coeff: (in_dim,out_dim,G+k)
    spline=B_batch(x_inputs,grids,k)
    y=tf.einsum('ijk,jlk->ijl',spline,coeff)
    return y


if __name__=='__main__':
    """
    test block
    """
    grid_range,num,k=[-1,1],8,3
    in_dim,out_dim=6,5


    delta_h=(grid_range[-1]-grid_range[0])/num
    grids=tf.cast(tf.linspace(grid_range[0]-delta_h*k,grid_range[-1]+delta_h*k,num+1+2*k),GLOBAL_TF_FLOAT_PRECISION)
    grids=tf.expand_dims(grids,axis=0)
    grids=tf.tile(grids,[in_dim,1])

    noise=tf.random.uniform([num+1,in_dim,out_dim],-1,1,dtype=GLOBAL_TF_FLOAT_PRECISION)*0.1
    coeff=curve2coeff(tf.transpose(grids,perm=[1,0])[k:-k,:],noise,grids,k)

    noise_pred=coeff2curve(tf.transpose(grids,perm=[1,0])[k:-k,:],grids,k,coeff)

    print(noise)
    print(noise-noise_pred)
