import numpy as np

from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.common import get_precision
from deepmd.utils.grid_based import curve2coeff,coeff2curve,rbf_coeff2curve,relu_coeff2curve
from deepmd.utils.fourier import f_coeff2curve
from deepmd.utils.chebyshev import cheb1_coeff2curve, cheb2_coeff2curve

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
                  scale_base=1.0,
                  bias_function='silu',
                  scale_bias_sigma=1.0,
                  scale_bias_miu=0.0,
                  degree=3
                  ):
    # For good accuracy, the last layer of the fitting network uses a higher precision neuron network.
    #bavg: the atomic energy shift
    if mixed_prec is not None and final_layer:
        inputs = tf.cast(inputs, get_precision(mixed_prec['output_prec']))
    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
        #scale_base and scale_bias are global parameters for all types of base functions
        if initial_variables is not None:
            scale_base_ini=tf.constant_initializer(initial_variables[name+'/scale_base'])
            scale_bias_ini=tf.constant_initializer(initial_variables[name+'/scale_bias'])
        else:
            scale_base_t=tf.ones([shape[1],outputs_size],dtype=precision)*scale_base
            scale_bias_t=scale_bias_sigma*tf.random.uniform([shape[1],outputs_size],-1,1,dtype=precision)/tf.sqrt(tf.constant(shape[1],dtype=precision))+scale_bias_miu/tf.sqrt(tf.constant(shape[1],dtype=precision))
        with tf.Session() as sess:
            scale_bias_ini=tf.constant_initializer(scale_bias_t.eval())
            scale_base_ini=tf.constant_initializer(scale_base_t.eval())

        #the grid based base function (fixed grids)
        if base_function in ['b_spline','rbf']:
            #map inputs into grid_range 
            g_low,g_high=grid_range
            inputs=0.5*(g_high-g_low)*tf.tanh(inputs)+0.5*(g_low+g_high)
            #initialization of grids and coeff
            delta_l=(grid_range[-1]-grid_range[0])/num
            if initial_variables is not None:
                grids_ini=tf.constant_initializer(initial_variables[name + '/grids'])
                coeff_ini=tf.constant_initializer(initial_variables[name + '/coeff'])
            else:
                #the global grids for each layer
                #shape (in_dim, num+2k+1)
                grids_t=tf.linspace(grid_range[0]-k*delta_l,grid_range[-1]+delta_l*k,num+1+2*k)
                grids_t=tf.cast(grids_t,precision)
                grids_t=tf.tile(tf.expand_dims(grids_t,0),[shape[1],1])
                if base_function=='b_spline':
                    #noise: (n_batch, in_dims, out_dim)
                    noise=noise_scale*tf.random.uniform([num+1,shape[1],outputs_size],-0.5,0.5,precision)/num
                    coeff_t=curve2coeff(tf.transpose(grids_t)[k:-k,:],noise,grids_t,k)
                elif base_function=='rbf':
                    coeff_t=noise_scale*tf.random.uniform([shape[1],outputs_size,num+1+2*k],-0.5,0.5,precision)/num
                else:
                    pass
                with tf.Session() as sess:
                    grids_ini=tf.constant_initializer(grids_t.eval())
                    coeff_ini=tf.constant_initializer(coeff_t.eval())
            #create trainable variables
            grids=tf.get_variable('grids',
                                  initializer=grids_ini(grids_t.shape,dtype=precision),
                                  trainable=False)
            variable_summaries(grids, 'grids')
            coeff=tf.get_variable('coeff',
                                  initializer=coeff_ini(coeff_t.shape,dtype=precision),
                                  trainable=True)
            variable_summaries(coeff, 'coeff')
            scale_base=tf.get_variable('scale_base',
                                       initializer=scale_base_ini(scale_base_t.shape,dtype=precision),
                                       trainable=base_trainable)
            variable_summaries(scale_base, 'scale_base')
            scale_bias=tf.get_variable('scale_bias',
                                       initializer=scale_bias_ini(scale_bias_t.shape,dtype=precision),
                                       trainable=bias_trainable)
            variable_summaries(scale_bias, 'scale_bias')
            #forward propagation
            if mixed_prec is not None and not final_layer:
                inputs=tf.cast(inputs,get_precision(mixed_prec['compute_prec']))
                grids=tf.cast(grids,get_precision(mixed_prec['compute_prec']))
                coeff=tf.cast(coeff,get_precision(mixed_prec['compute_prec']))
                scale_bias=tf.cast(scale_bias,get_precision(mixed_prec['compute_prec']))
                scale_base=tf.cast(scale_base,get_precision(mixed_prec['compute_prec']))
            if base_function=='b_spline':
                hidden_base=coeff2curve(inputs,grids,k,coeff)*scale_base
            elif base_function=='rbf':
                hidden_base=rbf_coeff2curve(inputs,grids,coeff,delta_l)*scale_base
            else:
                pass
        #the grid based base function (trainable grids)
        elif base_function in ['relu']:
            #the new input parameter degree is needed
            #map inputs into grid_range 
            g_low,g_high=grid_range
            inputs=0.5*(g_high-g_low)*tf.tanh(inputs)+0.5*(g_low+g_high)
            #initialization of grids and coeff
            delta_l=(grid_range[-1]-grid_range[0])/num
            if initial_variables is not None:
                grids_s_ini=tf.constant_initializer(initial_variables[name + '/grids_s'])
                grids_e_ini=tf.constant_initializer(initial_variables[name + '/grids_e'])
                coeff_ini=tf.constant_initializer(initial_variables[name + '/coeff'])
            else:
                #the global grids for each layer
                #shape (in_dim, num+2k+1)
                grids_s_t=tf.linspace(grid_range[0]-k*delta_l,grid_range[-1]-delta_l,num+k)
                grids_e_t=tf.linspace(grid_range[0]+delta_l,grid_range[-1]+delta_l*k,num+k)
                grids_s_t=tf.cast(grids_s_t,precision)
                grids_e_t=tf.cast(grids_e_t,precision)
                grids_s_t=tf.tile(tf.expand_dims(grids_s_t,0),[shape[1],1])
                grids_e_t=tf.tile(tf.expand_dims(grids_e_t,0),[shape[1],1])
                coeff_t=noise_scale*tf.random.uniform([shape[1],outputs_size,num+k],-0.5,0.5,precision)/num
                #coeff: (n_of_base_func or num+k, in_dims, out_dim)
                with tf.Session() as sess:
                    grids_s_ini=tf.constant_initializer(grids_s_t.eval())
                    grids_e_ini=tf.constant_initializer(grids_e_t.eval())
                    coeff_ini=tf.constant_initializer(coeff_t.eval())
            #create trainable variables
            grids_s=tf.get_variable('grids_s',
                                  initializer=grids_s_ini(grids_s_t.shape,dtype=precision),
                                  trainable=True)
            variable_summaries(grids_s, 'grids_s')
            grids_e=tf.get_variable('grids_e',
                                  initializer=grids_e_ini(grids_e_t.shape,dtype=precision),
                                  trainable=True)
            variable_summaries(grids_e, 'grids_e')
            coeff=tf.get_variable('coeff',
                                  initializer=coeff_ini(coeff_t.shape,dtype=precision),
                                  trainable=True)
            variable_summaries(coeff, 'coeff')
            scale_base=tf.get_variable('scale_base',
                                       initializer=scale_base_ini(scale_base_t.shape,dtype=precision),
                                       trainable=base_trainable)
            variable_summaries(scale_base, 'scale_base')
            scale_bias=tf.get_variable('scale_bias',
                                       initializer=scale_bias_ini(scale_bias_t.shape,dtype=precision),
                                       trainable=bias_trainable)
            variable_summaries(scale_bias, 'scale_bias')
            #forward propagation
            if mixed_prec is not None and not final_layer:
                inputs=tf.cast(inputs,get_precision(mixed_prec['compute_prec']))
                grids_s=tf.cast(grids_s,get_precision(mixed_prec['compute_prec']))
                grids_e=tf.cast(grids_e,get_precision(mixed_prec['compute_prec']))
                coeff=tf.cast(coeff,get_precision(mixed_prec['compute_prec']))
                scale_bias=tf.cast(scale_bias,get_precision(mixed_prec['compute_prec']))
                scale_base=tf.cast(scale_base,get_precision(mixed_prec['compute_prec']))
            hidden_base=relu_coeff2curve(inputs,grids_s,grids_e,coeff,delta_l*k,degree)*scale_base
        elif base_function in ['segment']:
            #map inputs into (-1, 1)
            inputs=tf.tanh(inputs)
            #create variable initializer
            if initial_variables is not None:
                coeff_ini=tf.constant_initializer(initial_variables[name + '/coeff'])
            else:
                coeff_t=noise_scale*tf.random.uniform([shape[1],outputs_size,num+1],-0.5,0.5,precision)/num
                with tf.Session() as sess:
                    coeff_ini=tf.constant_initializer(coeff_t.eval())

            #create trainable variables
            #coeff: (in, out, num+1)
            coeff=tf.get_variable('coeff',
                                   initializer=coeff_ini(coeff_t.shape,dtype=precision),
                                   trainable=True)
            variable_summaries(coeff, 'coeff')
            scale_base=tf.get_variable('scale_base',
                                       initializer=scale_base_ini(scale_base_t.shape,dtype=precision),
                                       trainable=base_trainable)
            variable_summaries(scale_base, 'scale_base')
            scale_bias=tf.get_variable('scale_bias',
                                       initializer=scale_bias_ini(scale_bias_t.shape,dtype=precision),
                                       trainable=bias_trainable)
            variable_summaries(scale_bias, 'scale_bias')
            #forward propagation
            if mixed_prec is not None and not final_layer:
                inputs=tf.cast(inputs,get_precision(mixed_prec['compute_prec']))
                coeff=tf.cast(coeff,get_precision(mixed_prec['compute_prec']))
                scale_bias=tf.cast(scale_bias,get_precision(mixed_prec['compute_prec']))
                scale_base=tf.cast(scale_base,get_precision(mixed_prec['compute_prec']))
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
                scales=tf.squeeze(scales)
                indices_h=tf.concat([mesh,seg_idx_h],axis=-1)
                sel_hidden_h=tf.gether_nd(coeff,indices_h)
                hidden_base=hidden_base*(tf.cast(1.0,scales.dtype)-scales)+sel_hidden_h*scales
            hidden_base=tf.einsum('ijk,ij->ijk',hidden_base,inputs)

        elif base_function=='fourier':
            #map inputs into [-pi, pi]
            inputs=tf.constant(3.1415926535,dtype=precision)*tf.tanh(inputs)
            #create variable initializer
            if initial_variables is not None:
                coeff_alpha_ini=tf.constant_initializer(initial_variables[name + '/coeff_alpha'])
                coeff_beta_ini=tf.constant_initializer(initial_variables[name + '/coeff_beta'])
            else:
                coeff_alpha_t=noise_scale*tf.random.uniform([shape[1],outputs_size,num],-0.5,0.5,precision)/num
                coeff_beta_t=noise_scale*tf.random.uniform([shape[1],outputs_size,num],-0.5,0.5,precision)/num
                with tf.Session() as sess:
                    coeff_alpha_ini=tf.constant_initializer(coeff_alpha_t.eval())
                    coeff_beta_ini=tf.constant_initializer(coeff_beta_t.eval())
            #create trainable variables
            coeff_alpha=tf.get_variable('coeff_alpha',
                                        initializer=coeff_alpha_ini(coeff_alpha_t.shape,dtype=precision),
                                        trainable=True)
            variable_summaries(coeff_alpha, 'coeff_alpha')
            coeff_beta=tf.get_variable('coeff_beta',
                                        initializer=coeff_beta_ini(coeff_beta_t.shape,dtype=precision),
                                        trainable=True)
            variable_summaries(coeff_beta, 'coeff_beta')
            scale_base=tf.get_variable('scale_base',
                                       initializer=scale_base_ini(scale_base_t.shape,dtype=precision),
                                       trainable=base_trainable)
            variable_summaries(scale_base, 'scale_base')
            scale_bias=tf.get_variable('scale_bias',
                                       initializer=scale_bias_ini(scale_bias_t.shape,dtype=precision),
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
        elif base_function in ['chebyshev1','chebyshev2']:
            #map inputs into [-1, 1]
            inputs=tf.tanh(inputs)
            #create variable initializer
            #the num parameter is the degree of chebyshev polynomial
            if initial_variables is not None:
                coeff_ini=tf.constant_initializer(initial_variables[name + '/coeff'])
            else:
                coeff_t=noise_scale*tf.random.uniform([shape[1],outputs_size,num+1],-0.5,0.5,precision)/num #degree+1=num
                with tf.Session() as sess:
                    coeff_ini=tf.constant_initializer(coeff_t.eval())
            #create trainable variables
            coeff=tf.get_variable('coeff',
                                        initializer=coeff_ini(coeff_t.shape,dtype=precision),
                                        trainable=True)
            variable_summaries(coeff, 'coeff')
            scale_base=tf.get_variable('scale_base',
                                       initializer=scale_base_ini(scale_base_t.shape,dtype=precision),
                                       trainable=base_trainable)
            variable_summaries(scale_base, 'scale_base')
            scale_bias=tf.get_variable('scale_bias',
                                       initializer=scale_bias_ini(scale_bias_t.shape,dtype=precision),
                                       trainable=bias_trainable)
            variable_summaries(scale_bias, 'scale_bias')
            #forward propagation
            if mixed_prec is not None and not final_layer:
                inputs=tf.cast(inputs,get_precision(mixed_prec['compute_prec']))
                coeff=tf.cast(coeff,get_precision(mixed_prec['compute_prec']))
                scale_bias=tf.cast(scale_bias,get_precision(mixed_prec['compute_prec']))
                scale_base=tf.cast(scale_base,get_precision(mixed_prec['compute_prec']))
            if base_function=='chebyshev1':
                hidden_base=cheb1_coeff2curve(inputs,coeff)*scale_base
            else:
                hidden_base=cheb2_coeff2curve(inputs,coeff)*scale_base
        elif base_function in ['skan-soft','skan-lsin','skan-lcos','skan-larctan','wav-mexican','wav-morlet','wav-dog','wav-shannon']:
            #create variable initializer
            if initial_variables is not None:
                coeff_ini=tf.constant_initializer(initial_variables[name + '/coeff'])
                if 'wav' in base_function:
                    coeff2_ini=tf.constant_initializer(initial_variables[name + '/coeff2'])
            else:
                coeff_t=noise_scale*tf.random.uniform([shape[1],outputs_size],-0.5,0.5,precision)/num 
                if 'wav' in base_function:
                    coeff2_t=noise_scale*tf.random.uniform([shape[1],outputs_size],0.5,1,precision)/num 
                with tf.Session() as sess:
                    coeff_ini=tf.constant_initializer(coeff_t.eval())
                    if 'wav' in base_function:
                        coeff2_ini=tf.constant_initializer(coeff2_t.eval())
            #create trainable variables
            coeff=tf.get_variable('coeff',
                                   initializer=coeff_ini(coeff_t.shape,dtype=precision),
                                   trainable=True)
            variable_summaries(coeff, 'coeff')
            if 'wav' in base_function:
                coeff2=tf.get_variable('coeff2',
                                       initializer=coeff2_ini(coeff2_t.shape,dtype=precision),
                                       trainable=True)
                variable_summaries(coeff, 'coeff2')
            scale_base=tf.get_variable('scale_base',
                                       initializer=scale_base_ini(scale_base_t.shape,dtype=precision),
                                       trainable=base_trainable)
            variable_summaries(scale_base, 'scale_base')
            scale_bias=tf.get_variable('scale_bias',
                                       initializer=scale_bias_ini(scale_bias_t.shape,dtype=precision),
                                       trainable=bias_trainable)
            variable_summaries(scale_bias, 'scale_bias')
            #forward propagation
            if mixed_prec is not None and not final_layer:
                inputs=tf.cast(inputs,get_precision(mixed_prec['compute_prec']))
                coeff=tf.cast(coeff,get_precision(mixed_prec['compute_prec']))
                if 'wav' in base_function:
                    coeff2=tf.cast(coeff2,get_precision(mixed_prec['compute_prec']))
                scale_bias=tf.cast(scale_bias,get_precision(mixed_prec['compute_prec']))
                scale_base=tf.cast(scale_base,get_precision(mixed_prec['compute_prec']))
            coeff=tf.expand_dims(coeff,axis=0)#(1,in_dim,out_dim)
            if 'wav' in base_function:
                coeff2=tf.expand_dims(coeff2,axis=0)#(1,in_dim,out_dim)
            xs=tf.expand_dims(inputs,axis=-1)#(batch,in_dim,1)
            if base_function=='skan-soft':
                hidden_base=tf.log(tf.exp(coeff*xs))-tf.cast(tf.log(2.0),precision) #(batch,in_dim,out_dim)
            elif base_function=='skan-lsin':
                hidden_base=coeff*tf.sin(xs)
            elif base_function=='skan-lcos':
                hidden_base=coeff*tf.cos(xs)
            elif base_function=='skan-larctan':
                hidden_base=tf.atan(coeff*xs)
            elif base_function=='wav-mexican':
                t=(xs-coeff)/coeff2
                hidden_base=tf.cast(2.0,precision)*(tf.square(t)-tf.cast(1.0,precision))*tf.exp(-tf.square(t)/tf.cast(2.0,precision))/tf.cast(tf.sqrt(3.0),precision)*tf.cast(tf.pow(3.14159,0.25),precision)
            elif base_function=='wav-morlet':
                t=(xs-coeff)/coeff2
                hidden_base=tf.cast(tf.cos(tf.cast(5.0,precision)*t)*tf.exp(-tf.square(t)/tf.cast(2.0,precision)),precision)
            elif base_function=='wav-dog':
                t=(xs-coeff)/coeff2
                hidden_base=tf.cast(t*tf.exp(-tf.square(t)/2.0),precision)
            elif base_function=='wav-shannon':
                t=(xs-coeff)/coeff2
                tf.cast(tf.experimental.numpy.sinc(t/tf.cast(3.1415926,precision))*tf.exp(-tf.squre(t)/tf.cast(2.0,precision)),precision)
            else:
                pass
            #hidden_base: (batch, in, out)
        else:
            pass

        if bias_function=='silu':
            #hidden_bias: (batch, in ,out)
            x=tf.tile(tf.expand_dims(inputs,axis=-1),[1,1,outputs_size])
            hidden_bias=tf.nn.silu(x)*scale_bias
        hidden=tf.reduce_sum(hidden_base+hidden_bias,axis=-2)
        return hidden+bavg


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


