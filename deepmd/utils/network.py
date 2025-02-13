import numpy as np

from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.common import get_precision
from deepmd.utils.bspline import curve2coeff,coeff2curve
from deepmd.utils.fourier import f_coeff2curve

def one_layer_rand_seed_shift():
    return 3

def one_kan_layer(inputs, 
                  outputs_size, 
                  precision = GLOBAL_TF_FLOAT_PRECISION, 
                  name='linear', 
                  reuse=None,
                  initial_variables = None,
                  mixed_prec = None,
                  final_layer = False,
                  bavg=0.0,
                  bias_trainable=True,
                  base_trainable=True,

                  base_function='b_spline',
                  k=3,
                  grid_range=[-1,1],
                  num=5,
                  noise_scale=0.1,
                  bias_function=tf.nn.silu,
                  scale_bias_sigma=1.0,
                  scale_bias_miu=0.0,
                  scale_base_sigma=1.0,
                  scale_base_miu=0.0,
                  degree=3
                  ):
    # For good accuracy, the last layer of the fitting network uses a higher precision neuron network.
    #bavg: the atomic energy shift
    if mixed_prec is not None and final_layer:
        inputs = tf.cast(inputs, get_precision(mixed_prec['output_prec']))

    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
        #the grid based base function (fixed grids)
        if base_function=='b_spline':
            #map inputs into grid_range 
            g_low,g_high=grid_range
            inputs=0.5*(g_high-g_low)*tf.tanh(inputs)+0.5*(g_low+g_high)
            #initialization of grids and coeff
            delta_l=(grid_range[-1]-grid_range[0])/num

            if initial_variables is not None:
                grids_ini=tf.constant_initializer(initial_variables[name + '/grids'])
                coeff_ini=tf.constant_initializer(initial_variables[name + '/coeff'])
                scale_base_ini=tf.constant_initializer(initial_variables[name+'/scale_base'])
                scale_bias_ini=tf.constant_initializer(initial_variables[name+'/scale_bias'])
            else:
                #the global grids for each layer
                #shape (in_dim, num+2k+1)
                grids_t=tf.linspace(grid_range[0]-k*delta_l,grid_range[-1]+delta_l*k,num+1+2*k)
                grids_t=tf.cast(grids_t,precision)
                grids_t=tf.tile(tf.expand_dims(grids_t,0),[shape[1],1])
                #noise: (n_batch, in_dims, out_dim)
                noise=noise_scale*tf.random.uniform([num+1,shape[1],outputs_size],-0.5,0.5,precision)/num
                coeff_t=curve2coeff(tf.transpose(grids_t)[k:-k,:],noise,grids_t,k)
                with tf.Session() as sess:
                    grids_ini=tf.constant_initializer(grids_t.eval())
                    coeff_ini=tf.constant_initializer(coeff_t.eval())
                scale_base_ini=tf.random_normal_initializer(stddev=scale_base_sigma,mean=scale_base_miu)
                scale_bias_ini=tf.random_normal_initializer(stddev=scale_bias_sigma,mean=scale_bias_miu)

            #create trainable variables
            grids=tf.get_variable('grids',
                                  [shape[-1],num+1+2*k],
                                  precision,
                                  grids_ini,
                                  trainable=False)
            variable_summaries(grids, 'grids')
            coeff=tf.get_variable('coeff',
                                  [shape[-1],outputs_size,num+k],
                                  precision,
                                  coeff_ini,
                                  trainable=True)
            variable_summaries(coeff, 'coeff')
            scale_base=tf.get_variable('scale_base',
                                       [shape[-1],outputs_size],
                                       precision,
                                       scale_base_ini,
                                       trainable=base_trainable)
            variable_summaries(scale_base, 'scale_base')
            scale_bias=tf.get_variable('scale_bias',
                                       [shape[-1],outputs_size],
                                       precision,
                                       scale_bias_ini,
                                       trainable=bias_trainable)
            variable_summaries(scale_bias, 'scale_bias')
            #forward propagation
            if mixed_prec is not None and not final_layer:
                inputs=tf.cast(inputs,get_precision(mixed_prec['compute_prec']))
                grids=tf.cast(grids,get_precision(mixed_prec['compute_prec']))
                coeff=tf.cast(coeff,get_precision(mixed_prec['compute_prec']))
                scale_bias=tf.cast(scale_bias,get_precision(mixed_prec['compute_prec']))
                scale_base=tf.cast(scale_base,get_precision(mixed_prec['compute_prec']))

            hidden_base=coeff2curve(inputs,grids,k,coeff)*scale_base
            #hidden_bias: (batch, in ,out)
            hidden_bias=tf.nn.silu(tf.expand_dims(inputs,axis=-1))*scale_bias
            hidden=tf.reduce_sum(hidden_base+hidden_bias,axis=-2)
            return hidden+bavg
        elif base_function=='fourier':
            #map inputs into [-pi, pi]
            inputs=tf.constant(3.1415926535,dtype=precision)*tf.tanh(inputs)
            #create variable initializer
            if initial_variables is not None:
                coeff_alpha_ini=tf.constant_initializer(initial_variables[name + '/coeff_alpha'])
                coeff_beta_ini=tf.constant_initializer(initial_variables[name + '/coeff_beta'])
                scale_base_ini=tf.constant_initializer(initial_variables[name+'/scale_base'])
                scale_bias_ini=tf.constant_initializer(initial_variables[name+'/scale_bias'])
            else:
                coeff_alpha_ini=tf.random_normal_initializer(stddev=scale_base_sigma,mean=scale_base_miu)
                coeff_beta_ini=tf.random_normal_initializer(stddev=scale_base_sigma,mean=scale_base_miu)
                scale_base_ini=tf.random_normal_initializer(stddev=scale_base_sigma,mean=scale_base_miu)
                scale_bias_ini=tf.random_normal_initializer(stddev=scale_bias_sigma,mean=scale_bias_miu)
            #create trainable variables
            coeff_alpha=tf.get_variable('coeff_alpha',
                                        [shape[-1],outputs_size,num],
                                        precision,
                                        coeff_alpha_ini,
                                        trainable=True)
            variable_summaries(coeff_alpha, 'coeff_alpha')
            coeff_beta=tf.get_variable('coeff_beta',
                                       [shape[-1],outputs_size,num],
                                       precision,
                                       coeff_beta_ini,
                                       trainable=True)
            variable_summaries(coeff_beta, 'coeff_beta')
            scale_base=tf.get_variable('scale_base',
                                       [shape[-1],outputs_size],
                                       precision,
                                       initializer=scale_base_ini,
                                       trainable=base_trainable)
            variable_summaries(scale_base, 'scale_base')
            scale_bias=tf.get_variable('scale_bias',
                                       [shape[-1],outputs_size],
                                       precision,
                                       scale_bias_ini,
                                       trainable=bias_trainable)
            variable_summaries(scale_bias, 'scale_bias')
            #forward propagation
            if mixed_prec is not None and not final_layer:
                inputs=tf.cast(inputs,get_precision(mixed_prec['compute_prec']))
                coeff_alpha=tf.cast(coeff_alpha,get_precision(mixed_prec['compute_prec']))
                coeff_beta=tf.cast(coeff_beta,get_precision(mixed_prec['compute_prec']))
                scale_bias=tf.cast(scale_bias,get_precision(mixed_prec['compute_prec']))
                scale_base=tf.cast(scale_base,get_precision(mixed_prec['compute_prec']))
            hidden_base=f_coeff2curve(inputs,coeff_alpha,coeff_beta)*scale_base
            return tf.reduce_sum(hidden_base,axis=-2)+bavg
        elif base_function=='segment':
            #map inputs within (-1, 1)
            inputs=tf.tanh(inputs)
            #scale the inputs to span across (-1, 1) to prevent aggregation
            #in_max, in_min, inputs_span, smalls, denominator: (batch, in)
            in_max=tf.math.reduce_max(inputs,axis=-1,keepdims=True)
            in_min=tf.math.reduce_min(inputs,axis=-1,keepdims=True)
            inputs_span=in_max-in_min
            smalls=tf.ones_like(inputs_span,dtype=inputs.dtype)*tf.cast(1e-6,inputs.dtype)
            denominator=tf.where(inputs_span==0.0,smalls,inputs_span)
            inputs=tf.cast(2.0-2e-6,precision)*((inputs-in_min)/(denominator)-tf.cast(0.5,precision))
            #create variable initializer
            if initial_variables is not None:
                coeff_ini=tf.constant_initializer(initial_variables[name + '/coeff'])
            else:
                coeff_ini=tf.random_normal_initializer(stddev=scale_base_sigma/np.sqrt(shape[-1]+outputs_size),mean=scale_base_miu)
            #create trainable variables
            #coeff: (in, out, num+1)
            coeff=tf.get_variable('coeff',
                                  [shape[-1],outputs_size,num+1],
                                  precision,
                                  coeff_ini,
                                  trainable=True)
            variable_summaries(coeff, 'coeff')

            #forward propagation
            if mixed_prec is not None and not final_layer:
                inputs=tf.cast(inputs,get_precision(mixed_prec['compute_prec']))
                coeff=tf.cast(coeff,get_precision(mixed_prec['compute_prec']))
            delta_l=tf.cast(2.0,inputs.dtype)/tf.cast(num,inputs.dtype)
            #xs: (batch, in, out)
            xs=tf.tile(tf.expand_dims(inputs,axis=-1),[1,1,outputs_size])-tf.cast(-1.0,inputs.dtype)
            #seg_idx_l, seg_idx_h: (batch, in, out, 1)
            seg_idx_l=tf.expand_dims(tf.cast(tf.math.floordiv(xs,delta_l),tf.int32),axis=-1)
            seg_idx_h=seg_idx_l+tf.cast(1,tf.int32)
            #scales: (batch, in, out, 1)
            scales=tf.expand_dims(tf.math.floormod(xs,delta_l),axis=-1)
            #zeros: (batch, in, out, 1)
            zeros=tf.zeros_like(scales,dtype=seg_idx_l.dtype)
            outs,ins=tf.range(0,outputs_size),tf.range(0,shape[1])
            #OUT, IN: (in, out)
            OUT,IN=tf.meshgrid(outs,ins)
            #mesh: (1, in, out, 2)
            mesh=tf.cast(tf.expand_dims(tf.stack([IN,OUT],axis=-1),axis=0),seg_idx_l.dtype)
            #mesh: (batch, in, out, 2)
            mesh=mesh+zeros
            #indices_l: (batch, in, out, 3)
            indices_l=tf.concat([mesh,seg_idx_l],axis=-1)
            #hidden_base: (batch, in, out)
            hidden_base=tf.gather_nd(coeff,indices_l)

            if degree>0:
                scales=tf.squeeze(scales,axis=-1)
                indices_h=tf.concat([mesh,seg_idx_h],axis=-1)
                sel_hidden_h=tf.gather_nd(coeff,indices_h)
                hidden_base=hidden_base*(tf.cast(1.0,scales.dtype)-scales)+sel_hidden_h*scales
            #hidden_base=tf.einsum('ijk,ij->ijk',hidden_base,inputs)
            hidden_base=hidden_base*tf.expand_dims(inputs,axis=-1)
            return tf.reduce_sum(hidden_base,axis=-2)+bavg
        else:
            pass


def one_layer(inputs, 
              outputs_size, 
              activation_fn=tf.nn.tanh, 
              precision = GLOBAL_TF_FLOAT_PRECISION, 
              stddev=1.0,
              bavg=0.0,
              name='linear', 
              reuse=None,
              seed=None, 
              use_timestep = False, 
              trainable = True,
              useBN = False, 
              uniform_seed = False,
              initial_variables = None,
              mixed_prec = None,
              final_layer = False):
    # For good accuracy, the last layer of the fitting network uses a higher precision neuron network.
    if mixed_prec is not None and final_layer:
        inputs = tf.cast(inputs, get_precision(mixed_prec['output_prec']))
    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
        w_initializer  = tf.random_normal_initializer(
                            stddev=stddev / np.sqrt(shape[1] + outputs_size),
                            seed=seed if (seed is None or uniform_seed) else seed + 0)
        b_initializer  = tf.random_normal_initializer(
                            stddev=stddev,
                            mean=bavg,
                            seed=seed if (seed is None or uniform_seed) else seed + 1)
        if initial_variables is not None:
            w_initializer = tf.constant_initializer(initial_variables[name + '/matrix'])
            b_initializer = tf.constant_initializer(initial_variables[name + '/bias'])
        w = tf.get_variable('matrix', 
                            [shape[1], outputs_size], 
                            precision,
                            w_initializer, 
                            trainable = trainable)
        variable_summaries(w, 'matrix')
        b = tf.get_variable('bias', 
                            [outputs_size], 
                            precision,
                            b_initializer, 
                            trainable = trainable)
        variable_summaries(b, 'bias')


        if mixed_prec is not None and not final_layer:
            inputs = tf.cast(inputs, get_precision(mixed_prec['compute_prec']))
            w = tf.cast(w, get_precision(mixed_prec['compute_prec']))
            b = tf.cast(b, get_precision(mixed_prec['compute_prec']))

        hidden = tf.matmul(inputs, w) + b
        if activation_fn != None and use_timestep :
            idt_initializer = tf.random_normal_initializer(
                                    stddev=0.001,
                                    mean=0.1,
                                    seed=seed if (seed is None or uniform_seed) else seed + 2)
            if initial_variables is not None:
                idt_initializer = tf.constant_initializer(initial_variables[name + '/idt'])
            idt = tf.get_variable('idt',
                                  [outputs_size],
                                  precision,
                                  idt_initializer, 
                                  trainable = trainable)
            variable_summaries(idt, 'idt')
        if activation_fn != None:
            if useBN:
                None
                # hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)   
                # return activation_fn(hidden_bn)
            else:
                if use_timestep :
                    if mixed_prec is not None and not final_layer:
                       idt = tf.cast(idt, get_precision(mixed_prec['compute_prec']))
                    return tf.reshape(activation_fn(hidden), [-1, outputs_size]) * idt
                else :
                    return tf.reshape(activation_fn(hidden), [-1, outputs_size])

        else:
            if useBN:
                None
                # return self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
            else:
                return hidden


def embedding_net_rand_seed_shift(
        network_size
):
    shift = 3 * (len(network_size) + 1)
    return shift

def embedding_net(xx,
                  network_size,
                  precision,
                  activation_fn = tf.nn.tanh,
                  resnet_dt = False,
                  name_suffix = '',
                  stddev = 1.0,
                  bavg = 0.0,
                  seed = None,
                  trainable = True, 
                  uniform_seed = False,
                  initial_variables = None,
                  mixed_prec = None):
    r"""The embedding network.

    The embedding network function :math:`\mathcal{N}` is constructed by is the
    composition of multiple layers :math:`\mathcal{L}^{(i)}`:

    .. math::
        \mathcal{N} = \mathcal{L}^{(n)} \circ \mathcal{L}^{(n-1)}
        \circ \cdots \circ \mathcal{L}^{(1)}

    A layer :math:`\mathcal{L}` is given by one of the following forms,
    depending on the number of nodes: [1]_

    .. math::
        \mathbf{y}=\mathcal{L}(\mathbf{x};\mathbf{w},\mathbf{b})=
        \begin{cases}
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b}) + \mathbf{x}, & N_2=N_1 \\
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b}) + (\mathbf{x}, \mathbf{x}), & N_2 = 2N_1\\
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b}), & \text{otherwise} \\
        \end{cases}

    where :math:`\mathbf{x} \in \mathbb{R}^{N_1}`$` is the input vector and :math:`\mathbf{y} \in \mathbb{R}^{N_2}`
    is the output vector. :math:`\mathbf{w} \in \mathbb{R}^{N_1 \times N_2}` and
    :math:`\mathbf{b} \in \mathbb{R}^{N_2}`$` are weights and biases, respectively,
    both of which are trainable if `trainable` is `True`. :math:`\boldsymbol{\phi}`
    is the activation function.

    Parameters
    ----------
    xx : Tensor   
        Input tensor :math:`\mathbf{x}` of shape [-1,1]
    network_size: list of int
        Size of the embedding network. For example [16,32,64]
    precision: 
        Precision of network weights. For example, tf.float64
    activation_fn:
        Activation function :math:`\boldsymbol{\phi}`
    resnet_dt: boolean
        Using time-step in the ResNet construction
    name_suffix: str
        The name suffix append to each variable. 
    stddev: float
        Standard deviation of initializing network parameters
    bavg: float
        Mean of network intial bias
    seed: int
        Random seed for initializing network parameters
    trainable: boolean
        If the network is trainable
    uniform_seed : boolean
        Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    initial_variables : dict
        The input dict which stores the embedding net variables
    mixed_prec
        The input dict which stores the mixed precision setting for the embedding net


    References
    ----------
    .. [1] Kaiming  He,  Xiangyu  Zhang,  Shaoqing  Ren,  and  Jian  Sun. Identitymappings
       in deep residual networks. InComputer Vision – ECCV 2016,pages 630–645. Springer
       International Publishing, 2016.
    """
    input_shape = xx.get_shape().as_list()
    outputs_size = [input_shape[1]] + network_size

    for ii in range(1, len(outputs_size)):
        w_initializer = tf.random_normal_initializer(
                            stddev=stddev/np.sqrt(outputs_size[ii]+outputs_size[ii-1]), 
                            seed = seed if (seed is None or uniform_seed)  else seed + ii*3+0
                        )
        b_initializer = tf.random_normal_initializer(
                            stddev=stddev, 
                            mean = bavg, 
                            seed = seed if (seed is None or uniform_seed) else seed + 3*ii+1
                        )
        if initial_variables is not None:
            scope = tf.get_variable_scope().name
            w_initializer = tf.constant_initializer(initial_variables[scope+'/matrix_'+str(ii)+name_suffix])
            b_initializer = tf.constant_initializer(initial_variables[scope+'/bias_'+str(ii)+name_suffix])
        w = tf.get_variable('matrix_'+str(ii)+name_suffix,
                            [outputs_size[ii - 1], outputs_size[ii]], 
                            precision,
                            w_initializer,
                            trainable = trainable)
        variable_summaries(w, 'matrix_'+str(ii)+name_suffix)

        b = tf.get_variable('bias_'+str(ii)+name_suffix, 
                            [1, outputs_size[ii]], 
                            precision,
                            b_initializer, 
                            trainable = trainable)
        variable_summaries(b, 'bias_'+str(ii)+name_suffix)

        if mixed_prec is not None:
            xx = tf.cast(xx, get_precision(mixed_prec['compute_prec']))
            w  = tf.cast(w,  get_precision(mixed_prec['compute_prec']))
            b  = tf.cast(b,  get_precision(mixed_prec['compute_prec']))
        hidden = tf.reshape(activation_fn(tf.matmul(xx, w) + b), [-1, outputs_size[ii]])
        if resnet_dt :
            idt_initializer = tf.random_normal_initializer(
                                  stddev=0.001, 
                                  mean = 1.0, 
                                  seed = seed if (seed is None or uniform_seed) else seed + 3*ii+2
                              )
            if initial_variables is not None:
                scope = tf.get_variable_scope().name
                idt_initializer = tf.constant_initializer(initial_variables[scope+'/idt_'+str(ii)+name_suffix])
            idt = tf.get_variable('idt_'+str(ii)+name_suffix, 
                                  [1, outputs_size[ii]], 
                                  precision,
                                  idt_initializer, 
                                  trainable = trainable)
            variable_summaries(idt, 'idt_'+str(ii)+name_suffix)
            if mixed_prec is not None:
                idt = tf.cast(idt, get_precision(mixed_prec['compute_prec']))

        if outputs_size[ii] == outputs_size[ii-1]:
            if resnet_dt :
                xx += hidden * idt
            else :
                xx += hidden
        elif outputs_size[ii] == outputs_size[ii-1] * 2: 
            if resnet_dt :
                xx = tf.concat([xx,xx], 1) + hidden * idt
            else :
                xx = tf.concat([xx,xx], 1) + hidden
        else:
            xx = hidden
    return xx

def variable_summaries(var: tf.Variable, name: str):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).

    Parameters
    ----------
    var : tf.Variable
        [description]
    name : str
        variable name
    """
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

if __name__=='__main__':
    in_dim,out_dim=50,1
    x_input=tf.random.uniform([1,in_dim],-1,1,dtype=GLOBAL_TF_FLOAT_PRECISION)
    y,coeff=one_kan_layer(x_input,out_dim,bias_trainable=False,base_trainable=False)
    trainables=tf.trainable_variables()
    #y,w=one_layer(x_input,out_dim)

    print(trainables)
    grads=tf.gradients(y,trainables)[0]
    print(grads)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print(y.eval())
        print(grads.eval())


