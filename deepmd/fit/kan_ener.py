import warnings
import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.common import add_data_requirement, get_precision, ACTIVATION_FN_DICT, PRECISION_DICT, docstring_parameter, cast_precision
from deepmd.utils.argcheck import list_to_doc
#load kan layer primitive
from deepmd.utils.network import one_kan_layer
from deepmd.utils.type_embed import embed_atom_type
from deepmd.utils.graph import get_fitting_net_variables, load_graph_def, get_tensor_by_name_from_graph
from deepmd.fit.fitting import Fitting

from deepmd.env import global_cvt_2_tf_float
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION


class KanEnerFitting (Fitting):
    r"""Fitting the energy of the system. The force and the virial can also be trained.

    The potential energy :math:`E` is a fitting network function of the descriptor :math:`\mathcal{D}`:

    .. math::
        E(\mathcal{D}) = \mathcal{L}^{(n)} \circ \mathcal{L}^{(n-1)}
        \circ \cdots \circ \mathcal{L}^{(1)} \circ \mathcal{L}^{(0)}

    The first :math:`n` hidden layers :math:`\mathcal{L}^{(0)}, \cdots, \mathcal{L}^{(n-1)}` are given by

    .. math::
        \mathbf{y}=\mathcal{L}(\mathbf{x};\mathbf{w},\mathbf{b})=
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b})

    where :math:`\mathbf{x} \in \mathbb{R}^{N_1}`$` is the input vector and :math:`\mathbf{y} \in \mathbb{R}^{N_2}`
    is the output vector. :math:`\mathbf{w} \in \mathbb{R}^{N_1 \times N_2}` and
    :math:`\mathbf{b} \in \mathbb{R}^{N_2}`$` are weights and biases, respectively,
    both of which are trainable if `trainable[i]` is `True`. :math:`\boldsymbol{\phi}`
    is the activation function.

    The output layer :math:`\mathcal{L}^{(n)}` is given by

    .. math::
        \mathbf{y}=\mathcal{L}^{(n)}(\mathbf{x};\mathbf{w},\mathbf{b})=
            \mathbf{x}^T\mathbf{w}+\mathbf{b}

    where :math:`\mathbf{x} \in \mathbb{R}^{N_{n-1}}`$` is the input vector and :math:`\mathbf{y} \in \mathbb{R}`
    is the output scalar. :math:`\mathbf{w} \in \mathbb{R}^{N_{n-1}}` and
    :math:`\mathbf{b} \in \mathbb{R}`$` are weights and bias, respectively,
    both of which are trainable if `trainable[n]` is `True`.

    Parameters
    ----------
    descrpt
            The descrptor :math:`\mathcal{D}`
    neuron
            Number of neurons :math:`N` in each hidden layer of the fitting net
    numb_fparam
            Number of frame parameter
    numb_aparam
            Number of atomic parameter
    rcond
            The condition number for the regression of atomic energy.
    tot_ener_zero
            Force the total energy to zero. Useful for the charge fitting.
    trainable
            If the weights of fitting net are trainable. 
            Suppose that we have :math:`N_l` hidden layers in the fitting net, 
            this list is of length :math:`N_l + 1`, specifying if the hidden layers and the output layer are trainable.
    atom_ener
            Specifying atomic energy contribution in vacuum. The `set_davg_zero` key in the descrptor should be set.
    precision
            The precision of the embedding net parameters. Supported options are {1}                
    """
    @docstring_parameter(list_to_doc(ACTIVATION_FN_DICT.keys()), list_to_doc(PRECISION_DICT.keys()))
    def __init__ (self, 
                  descrpt : tf.Tensor,
                  neuron : List[int] = [120,60,30],
                  numb_fparam : int = 0,
                  numb_aparam : int = 0,
                  rcond : float = 1e-3,
                  tot_ener_zero : bool = False,
                  atom_ener : List[float] = [],
                  precision : str = 'default',

                  base_function : str='b_spline',
                  k : int=3,
                  grid_range : List[float]=[-1.0,1.0],
                  num : int=5,
                  noise_scale : float=0.1,
                  scale_base : float=1.0,
                  bias_function : str='silu',
                  scale_bias_miu : float=0.0,
                  scale_bias_sigma : float=1.0,
                  bias_trainable : bool=True,
                  base_trainable : bool=True,
                  degree: int=2
    ) -> None:
        """
        Constructor
        Including the addational inputs for KAN network:
            input parameters within 'fitting_net' for kan:
            {
                'type':'ener_kan',
                'base_function':'str', (b_spline or wave_let)\
                'k':int, (spline order if base_function==b_spline)\
                'init_grid_range':[float,float], (if base_function==b_spline)\
                'neuron':[int,int,...],\

                'update_grid': bool,\
                ...
                ''
            }
            base_function:
                The base function of univariate activation function within kan network  Supported options are {'b_spline','wave_let','fourier'}

        """
        # model param (global)
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()

        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.n_neuron = neuron
        self.rcond = rcond
        self.tot_ener_zero = tot_ener_zero
        self.fitting_precision = get_precision(precision)

        self.atom_ener = []
        for at, ae in enumerate(atom_ener):
            if ae is not None:
                self.atom_ener.append(tf.constant(ae, self.fitting_precision, name = "atom_%d_ener" % at))
            else:
                self.atom_ener.append(None)
        self.useBN = False
        self.bias_atom_e = None
        # data requirement
        if self.numb_fparam > 0 :
            add_data_requirement('fparam', self.numb_fparam, atomic=False, must=True, high_prec=False)
            self.fparam_avg = None
            self.fparam_std = None
            self.fparam_inv_std = None
        if self.numb_aparam > 0:
            add_data_requirement('aparam', self.numb_aparam, atomic=True,  must=True, high_prec=False)
            self.aparam_avg = None
            self.aparam_std = None
            self.aparam_inv_std = None

        self.fitting_net_variables = None
        self.mixed_prec = None

        # set parameters for KAN network
        self.kan_param_dict={}
        self.kan_param_dict['base_function']=base_function
        #if self.kan_param_dict['base_function']=='b_spline':
        if True:
            self.kan_param_dict['k']=k
            self.kan_param_dict['grid_range']=grid_range
            self.kan_param_dict['num']=num
            self.kan_param_dict['noise_scale']=noise_scale
            self.kan_param_dict['scale_base']=scale_base
            self.kan_param_dict['bias_function']=bias_function
            self.kan_param_dict['scale_bias_miu']=scale_bias_miu
            self.kan_param_dict['scale_bias_sigma']=scale_bias_sigma
            self.kan_param_dict['bias_trainable']=bias_trainable 
            self.kan_param_dict['base_trainable']=base_trainable
            self.kan_param_dict['degree']=degree
            if self.kan_param_dict['bias_trainable'] is None:
                self.kan_param_dict['bias_trainable'] = [True for ii in range(len(self.n_neuron) + 1)]
            if type(self.kan_param_dict['bias_trainable']) is bool:
                self.kan_param_dict['bias_trainable'] = [self.kan_param_dict['bias_trainable']] * (len(self.n_neuron)+1)
            if self.kan_param_dict['base_trainable'] is None:
                self.kan_param_dict['base_trainable'] = [True for ii in range(len(self.n_neuron) + 1)]
            if type(self.kan_param_dict['base_trainable']) is bool:
                self.kan_param_dict['base_trainable'] = [self.kan_param_dict['base_trainable']] * (len(self.n_neuron)+1)
            assert(len(self.kan_param_dict['bias_trainable']) == len(self.n_neuron) + 1), 'length of bias_trainable should be that of n_neuron + 1'
            assert(len(self.kan_param_dict['base_trainable']) == len(self.n_neuron) + 1), 'length of base_trainable should be that of n_neuron + 1'


    def get_numb_fparam(self) -> int:
        """
        Get the number of frame parameters
        """
        return self.numb_fparam

    def get_numb_aparam(self) -> int:
        """
        Get the number of atomic parameters
        """
        return self.numb_fparam

    def compute_output_stats(self, 
                             all_stat: dict
    ) -> None:
        """
        Compute the ouput statistics

        Parameters
        ----------
        all_stat
                must have the following components:
                all_stat['energy'] of shape n_sys x n_batch x n_frame
                can be prepared by model.make_stat_input
        """
        self.bias_atom_e = self._compute_output_stats(all_stat, rcond = self.rcond)

    @classmethod
    def _compute_output_stats(self, all_stat, rcond = 1e-3):
        data = all_stat['energy']
        # data[sys_idx][batch_idx][frame_idx]
        sys_ener = np.array([])
        for ss in range(len(data)):
            sys_data = []
            for ii in range(len(data[ss])):
                for jj in range(len(data[ss][ii])):
                    sys_data.append(data[ss][ii][jj])
            sys_data = np.concatenate(sys_data)
            sys_ener = np.append(sys_ener, np.average(sys_data))
        data = all_stat['natoms_vec']
        sys_tynatom = np.array([])
        nsys = len(data)
        for ss in range(len(data)):
            sys_tynatom = np.append(sys_tynatom, data[ss][0].astype(np.float64))
        sys_tynatom = np.reshape(sys_tynatom, [nsys,-1])
        sys_tynatom = sys_tynatom[:,2:]
        energy_shift,resd,rank,s_value \
            = np.linalg.lstsq(sys_tynatom, sys_ener, rcond = rcond)
        return energy_shift    

    def compute_input_stats(self, 
                            all_stat : dict,
                            protection : float = 1e-2) -> None:
        """
        Compute the input statistics

        Parameters
        ----------
        all_stat
                if numb_fparam > 0 must have all_stat['fparam']
                if numb_aparam > 0 must have all_stat['aparam']
                can be prepared by model.make_stat_input
        protection
                Divided-by-zero protection
        """
        # stat fparam
        if self.numb_fparam > 0:
            cat_data = np.concatenate(all_stat['fparam'], axis = 0)
            cat_data = np.reshape(cat_data, [-1, self.numb_fparam])
            self.fparam_avg = np.average(cat_data, axis = 0)
            self.fparam_std = np.std(cat_data, axis = 0)
            for ii in range(self.fparam_std.size):
                if self.fparam_std[ii] < protection:
                    self.fparam_std[ii] = protection
            self.fparam_inv_std = 1./self.fparam_std
        # stat aparam
        if self.numb_aparam > 0:
            sys_sumv = []
            sys_sumv2 = []
            sys_sumn = []
            for ss_ in all_stat['aparam'] : 
                ss = np.reshape(ss_, [-1, self.numb_aparam])
                sys_sumv.append(np.sum(ss, axis = 0))
                sys_sumv2.append(np.sum(np.multiply(ss, ss), axis = 0))
                sys_sumn.append(ss.shape[0])
            sumv = np.sum(sys_sumv, axis = 0)
            sumv2 = np.sum(sys_sumv2, axis = 0)
            sumn = np.sum(sys_sumn)
            self.aparam_avg = (sumv)/sumn
            self.aparam_std = self._compute_std(sumv2, sumv, sumn)
            for ii in range(self.aparam_std.size):
                if self.aparam_std[ii] < protection:
                    self.aparam_std[ii] = protection
            self.aparam_inv_std = 1./self.aparam_std


    def _compute_std (self, sumv2, sumv, sumn) :
        return np.sqrt(sumv2/sumn - np.multiply(sumv/sumn, sumv/sumn))

    def _build_lower(
            self,
            start_index,
            natoms,
            inputs,
            fparam = None,
            aparam = None, 
            bias_atom_e = 0.0,
            suffix = '',
            reuse = None
    ):
        # cut-out inputs
        inputs_i = tf.slice (inputs,
                             [ 0, start_index*      self.dim_descrpt],
                             [-1, natoms* self.dim_descrpt] )
        inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
        layer = inputs_i
        if fparam is not None:
            ext_fparam = tf.tile(fparam, [1, natoms])
            ext_fparam = tf.reshape(ext_fparam, [-1, self.numb_fparam])
            ext_fparam = tf.cast(ext_fparam,self.fitting_precision)
            layer = tf.concat([layer, ext_fparam], axis = 1)
        if aparam is not None:
            ext_aparam = tf.slice(aparam, 
                                  [ 0, start_index      * self.numb_aparam],
                                  [-1, natoms * self.numb_aparam])
            ext_aparam = tf.reshape(ext_aparam, [-1, self.numb_aparam])
            ext_aparam = tf.cast(ext_aparam,self.fitting_precision)
            layer = tf.concat([layer, ext_aparam], axis = 1)


        params={k:v for k,v in self.kan_param_dict.items() if k not in ['base_trainable','bias_trainable']}
        for ii in range(0,len(self.n_neuron)):
            #hidden kan layer
            layer = one_kan_layer(
                layer,
                self.n_neuron[ii],
                precision = self.fitting_precision,
                name='layer_'+str(ii)+suffix,
                reuse=reuse,
                initial_variables = self.fitting_net_variables,
                mixed_prec = self.mixed_prec,
                final_layer = False,
                bavg=0.0,
                bias_trainable=self.kan_param_dict['bias_trainable'][ii],
                base_trainable=self.kan_param_dict['base_trainable'][ii],

                **params
                )
        #final kan layer
        final_layer = one_kan_layer(
            layer, 
            1, 
            precision = self.fitting_precision, 
            name='final_layer'+suffix, 
            reuse=reuse, 
            initial_variables = self.fitting_net_variables,
            mixed_prec = self.mixed_prec,
            final_layer = True,
            bavg=bias_atom_e,
            bias_trainable=self.kan_param_dict['bias_trainable'][ii],
            base_trainable=self.kan_param_dict['base_trainable'][ii],
            **params
            )

        return final_layer
            
            
    @cast_precision
    def build (self, 
               inputs : tf.Tensor,
               natoms : tf.Tensor,
               input_dict : dict = {},
               reuse : bool = None,
               suffix : str = '', 
    ) -> tf.Tensor:
        """
        Build the computational graph for fitting net

        Parameters
        ----------
        inputs
                The input descriptor
        input_dict
                Additional dict for inputs. 
                                
                if numb_fparam > 0, should have input_dict['fparam']
                if numb_aparam > 0, should have input_dict['aparam']
                
        natoms
                The number of atoms. This tensor has the length of Ntypes + 2
                natoms[0]: number of local atoms
                natoms[1]: total number of atoms held by this processor
                natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        reuse
                The weights in the networks should be reused when get the variable.
        suffix
                Name suffix to identify this descriptor

        Returns
        -------
        ener
                The system energy
        """
        bias_atom_e = self.bias_atom_e
        if self.numb_fparam > 0 and ( self.fparam_avg is None or self.fparam_inv_std is None ):
            raise RuntimeError('No data stat result. one should do data statisitic, before build')
        if self.numb_aparam > 0 and ( self.aparam_avg is None or self.aparam_inv_std is None ):
            raise RuntimeError('No data stat result. one should do data statisitic, before build')

        with tf.variable_scope('fitting_attr' + suffix, reuse = reuse) :
            t_dfparam = tf.constant(self.numb_fparam, 
                                    name = 'dfparam', 
                                    dtype = tf.int32)
            t_daparam = tf.constant(self.numb_aparam, 
                                    name = 'daparam', 
                                    dtype = tf.int32)
            if self.numb_fparam > 0: 
                t_fparam_avg = tf.get_variable('t_fparam_avg', 
                                               self.numb_fparam,
                                               dtype = GLOBAL_TF_FLOAT_PRECISION,
                                               trainable = False,
                                               initializer = tf.constant_initializer(self.fparam_avg))
                t_fparam_istd = tf.get_variable('t_fparam_istd', 
                                                self.numb_fparam,
                                                dtype = GLOBAL_TF_FLOAT_PRECISION,
                                                trainable = False,
                                                initializer = tf.constant_initializer(self.fparam_inv_std))
            if self.numb_aparam > 0: 
                t_aparam_avg = tf.get_variable('t_aparam_avg', 
                                               self.numb_aparam,
                                               dtype = GLOBAL_TF_FLOAT_PRECISION,
                                               trainable = False,
                                               initializer = tf.constant_initializer(self.aparam_avg))
                t_aparam_istd = tf.get_variable('t_aparam_istd', 
                                                self.numb_aparam,
                                                dtype = GLOBAL_TF_FLOAT_PRECISION,
                                                trainable = False,
                                                initializer = tf.constant_initializer(self.aparam_inv_std))
            
        inputs = tf.reshape(inputs, [-1, self.dim_descrpt * natoms[0]])
        if len(self.atom_ener):
            # only for atom_ener
            inputs_zero = tf.zeros_like(inputs, dtype=self.fitting_precision)
        

        if bias_atom_e is not None :
            assert(len(bias_atom_e) == self.ntypes)

        fparam = None
        aparam = None
        if self.numb_fparam > 0 :
            fparam = input_dict['fparam']
            fparam = tf.reshape(fparam, [-1, self.numb_fparam])
            fparam = (fparam - t_fparam_avg) * t_fparam_istd            
        if self.numb_aparam > 0 :
            aparam = input_dict['aparam']
            aparam = tf.reshape(aparam, [-1, self.numb_aparam])
            aparam = (aparam - t_aparam_avg) * t_aparam_istd
            aparam = tf.reshape(aparam, [-1, self.numb_aparam * natoms[0]])
            
        if input_dict is not None:
            type_embedding = input_dict.get('type_embedding', None)
        else:
            type_embedding = None
        if type_embedding is not None:
            atype_embed = embed_atom_type(self.ntypes, natoms, type_embedding)
            atype_embed = tf.tile(atype_embed,[tf.shape(inputs)[0],1])
        else:
            atype_embed = None

        if atype_embed is None:
            start_index = 0
            for type_i in range(self.ntypes):
                if bias_atom_e is None :
                    type_bias_ae = 0.0
                else :
                    type_bias_ae = bias_atom_e[type_i]
                final_layer = self._build_lower(
                    start_index, natoms[2+type_i], 
                    inputs, fparam, aparam, 
                    bias_atom_e=type_bias_ae, suffix='_type_'+str(type_i)+suffix, reuse=reuse
                )
                # concat the results
                if type_i < len(self.atom_ener) and self.atom_ener[type_i] is not None:                
                    zero_layer = self._build_lower(
                        start_index, natoms[2+type_i], 
                        inputs_zero, fparam, aparam, 
                        bias_atom_e=type_bias_ae, suffix='_type_'+str(type_i)+suffix, reuse=True
                    )
                    final_layer += self.atom_ener[type_i] - zero_layer
                final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms[2+type_i]])
                # concat the results
                if type_i == 0:
                    outs = final_layer
                else:
                    outs = tf.concat([outs, final_layer], axis = 1)
                start_index += natoms[2+type_i]
        # with type embedding
        else:
            if len(self.atom_ener) > 0:
                raise RuntimeError("setting atom_ener is not supported by type embedding")
            atype_embed = tf.cast(atype_embed, self.fitting_precision)
            type_shape = atype_embed.get_shape().as_list()
            inputs = tf.concat(
                [tf.reshape(inputs,[-1,self.dim_descrpt]),atype_embed],
                axis=1
            )
            self.dim_descrpt = self.dim_descrpt + type_shape[1]
            inputs = tf.reshape(inputs, [-1, self.dim_descrpt * natoms[0]])
            final_layer = self._build_lower(
                0, natoms[0], 
                inputs, fparam, aparam, 
                bias_atom_e=0.0, suffix=suffix, reuse=reuse
            )
            outs = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms[0]])

        if self.tot_ener_zero:
            force_tot_ener = 0.0
            outs = tf.reshape(outs, [-1, natoms[0]])
            outs_mean = tf.reshape(tf.reduce_mean(outs, axis = 1), [-1, 1])
            outs_mean = outs_mean - tf.ones_like(outs_mean, dtype = GLOBAL_TF_FLOAT_PRECISION) * (force_tot_ener/global_cvt_2_tf_float(natoms[0]))
            outs = outs - outs_mean
            outs = tf.reshape(outs, [-1])

        tf.summary.histogram('fitting_net_output', outs)
        return tf.reshape(outs, [-1])


    def init_variables(self,
                       model_file: str
    ) -> None:
        """
        Init the fitting net variables with the given frozen model

        Parameters
        ----------
        model_file : str
            The input frozen model file
        """
        self.fitting_net_variables = get_fitting_net_variables(model_file)


    def enable_compression(self,
                           model_file: str,
                           suffix: str = ""
    ) -> None:
        """
        Set the fitting net attributes from the frozen model_file when fparam or aparam is not zero

        Parameters
        ----------
        model_file : str
            The input frozen model file
        suffix : str, optional
                The suffix of the scope
        """
        if self.numb_fparam > 0 or self.numb_aparam > 0:
            graph, _ = load_graph_def(model_file)
        if self.numb_fparam > 0:
            self.fparam_avg = get_tensor_by_name_from_graph(graph, 'fitting_attr%s/t_fparam_avg' % suffix)
            self.fparam_inv_std = get_tensor_by_name_from_graph(graph, 'fitting_attr%s/t_fparam_istd' % suffix)
        if self.numb_aparam > 0:
            self.aparam_avg = get_tensor_by_name_from_graph(graph, 'fitting_attr%s/t_aparam_avg' % suffix)
            self.aparam_inv_std = get_tensor_by_name_from_graph(graph, 'fitting_attr%s/t_aparam_istd' % suffix)
 

    def enable_mixed_precision(self, mixed_prec: dict = None) -> None:
        """
        Reveive the mixed precision setting.

        Parameters
        ----------
        mixed_prec
                The mixed precision setting used in the embedding net
        """
        self.mixed_prec = mixed_prec
        self.fitting_precision = get_precision(mixed_prec['output_prec'])


