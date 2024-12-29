from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION

def B_batch(x,grids,k,epsilon=1e-8):
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
        #values=tf.where(
        #        (x>=grids[:,:,:-1] & x<grids[:,:,1:]),
        #        tf.cast(1.0, GLOBAL_TF_FLOAT_PRECISION),
        #        tf.cast(0.0, GLOBAL_TF_FLOAT_PRECISION)
        #        )

    else:
        B_kml=B_batch(x[:,:,0],grids[0],k-1)
        values=B_kml[:,:,:-1]*(x - grids[:,:,:-(k + 1)])/(grids[:,:,k:-1]-grids[:,:,:-(k+1)]+epsilon)+B_kml[:,:,1:]*(grids[:,:,k+1:]-x)/(grids[:,:,k+1:]-grids[:,:,1:-k]+epsilon)

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


def rbf_batch(x,grids,h):
    """
     it takes a batch of x sample vector, returns the values of rbf base functions for each xi
     x:
         2D tensor
             shape: (batch, in_dim)
     grids:
         2D tensor
             shape: (in_dim, n_of_bias)
         int
     return:
         3D tensor
             shape: (batch, in_dim, n_of_bias)
    """
    #x: (batch, in_dim, 1)
    x=tf.expand_dims(x,axis=-1)
    #grids: (1, in_dims, n_of_bias)
    grids=tf.expand_dims(grids,axis=0)
    return tf.exp(-(x-grids)**2/2/(h**2))

def rbf_coeff2curve(x_inputs,grids,coeff,h):
    """
    it tackes x_inputs and target y_out, return the fitted coeff of base functions
    the x_inputs specify the x samples for each spline function and each input dimension of the layer
    x_input:
        2D tensor
            shape: (batch, in_dim)
    grids:
        2D tensor
            shape: (in_dim, n_of_bias)
    coeff:
        3D tensor
            shape: (in_dim, out_dim, n_of_bias)
    return:
        3D tensor
            shape (batch, in_dim, out_dim)
    """
    #mat: (batch,in_dim,n_of_bias)
    #coeff: (in_dim,out_dim,n_of_bias)
    mat=rbf_batch(x_inputs,grids,h)
    y=tf.einsum('ijk,jlk->ijl',mat,coeff)
    return y

def relu_batch(x,grids_s,grids_e,interval,degree):
    """
     it takes a batch of x sample vector, returns the values of rbf base functions for each xi
     x:
         2D tensor
             shape: (batch, in_dim)
     grids_s,grids_e:
         2D tensor
             shape: (in_dim, n_of_bias)
     interval: 
         float
     degree:
         int
     return:
         3D tensor
             shape: (batch, in_dim, n_of_bias)
    """
    #x: (batch, in_dim, 1)
    x=tf.expand_dims(x,axis=-1)
    #grids_s, grids_e: (1, in_dims, n_of_bias)
    grids_s,girds_e=tf.expand_dims(grids_s,axis=0),tf.expand_dims(grids_e,axis=0)
    return tf.pow(tf.nn.relu(x-grids_s)*tf.nn.relu(grids_e-x),degree)*tf.cast(tf.pow(2/interval,2*degree),GLOBAL_TF_FLOAT_PRECISION)

def relu_coeff2curve(x_inputs,grids_s,grids_e,coeff,interval,degree):
    """
    it tackes x_inputs and target y_out, return the fitted coeff of base functions
    the x_inputs specify the x samples for each spline function and each input dimension of the layer
    x_input:
        2D tensor
            shape: (batch, in_dim)
    grids_s, grids_e:
        2D tensor
            shape: (in_dim, n_of_bias)
    coeff:
        3D tensor
            shape: (in_dim, out_dim, n_of_bias)
    degree:
        int
    return:
        3D tensor
            shape (batch, in_dim, out_dim)
    """
    #x: (batch, in_dim, 1)
    #mat: (batch, in_dim, n_of_bias)
    mat=relu_batch(x_inputs,grids_s,grids_e,interval,degree)
    y=tf.einsum('ijk,jlk->ijl',mat,coeff)
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
