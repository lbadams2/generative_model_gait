import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli, Normal


class BernoulliLayer:
    def __init__(self, n_units, dtype):
        self.n_units = n_units
        self._tf_dtype = dtype

    def init(self, batch_size, random_seed=None):
        return tf.random_uniform([batch_size, self.n_units], minval=0., maxval=1.,
                                 dtype=self._tf_dtype, seed=random_seed, name='bernoulli_init')

    def activation(self, x, b):
        return tf.nn.sigmoid(x + b)

    def _sample(self, means):
        return Bernoulli(probs=means)


class GaussianLayer:
    def __init__(self, sigma, n_units, dtype):
        self.n_units = n_units
        self._tf_dtype = dtype
        self.sigma = sigma # sigma is 1
        #_sigma_tmp = np.repeat(sigma, self.n_units)


    def init(self, batch_size, random_seed=None):
        t = tf.random_normal([batch_size, self.n_units],
                             dtype=self._tf_dtype, seed=random_seed)
        t = tf.multiply(t, self.sigma, name='gaussian_init')
        return t

    def activation(self, x, b):
        t = x * self.sigma + b
        return t

    def _sample(self, means):
        return Normal(loc=means, scale=tf.cast(self.sigma, dtype=self._tf_dtype))
        

class RBM:

    def __init__(v_layer_cls, sigma, h_layer_cls, learning_rate, epochs, batch_size):
        self.dropout = .2
        self.dbm_first = False
        self.dbm_last = False
        self._tf_dtype = 'float32'
        self._dbm_first = tf.constant(self.dbm_first, dtype=tf.bool, name='is_dbm_first')
        self._dbm_last = tf.constant(self.dbm_last, dtype=tf.bool, name='is_dbm_last')
        t = tf.constant(1., dtype=self._tf_dtype, name="1")
        t1 = tf.cast(self._dbm_first, dtype=self._tf_dtype) # 0
        self._propup_multiplier = tf.identity(tf.add(t1, t), name='propup_multiplier') # multiplier is 1
        t2 = tf.cast(self._dbm_last, dtype=self._tf_dtype) # 0
        self._propdown_multiplier = tf.identity(tf.add(t2, t), name='propdown_multiplier') # multiplier is 1
        self.sample_h_states = True
        self.n_gibbs_steps = []
        self.n_gibbs_steps.append(1) # could be [1,1,1]

        self.epochs = epochs
        self.batch_size = batch_size
        
        # normalization/regularization?
        self.l2 = 1e-4
        self._l2 = tf.constant(self.l2, dtype=self._tf_dtype, name='L2_coef')

        self.n_hidden = 3
        self.hb_init = 0
        hb_init = np.repeat(self.hb_init, self.n_hidden) # creates [0,0,0]
        hb_init = tf.constant(hb_init, dtype=self._tf_dtype, name='hb_init')
        self._hb = tf.Variable(hb_init, dtype=self._tf_dtype, name='hb')

        self.n_visible = 6
        self.vb_init = 0
        vb_init = np.repeat(self.vb_init, self.n_visible)
        vb_init = tf.constant(vb_init, dtype=self._tf_dtype, name='vb_init')
        self._vb = tf.Variable(vb_init, dtype=self._tf_dtype, name='vb')

        self._h_layer = h_layer_cls(self.n_hidden, 'float32')
        self._v_layer = v_layer_cls(sigma, self.n_visible, 'float32')
        
        W_init = tf.random_normal([self._n_visible, self._n_hidden],
                                           mean=0.0, stddev=self.W_init,
                                           seed=self.random_seed, dtype=self._tf_dtype)
        W_init = tf.identity(W_init, name='W_init') # unsure about this, could also set W_init to constant .01
        self._W = tf.Variable(W_init, dtype=self._tf_dtype, name='W')

        dW_init = tf.zeros([self._n_visible, self._n_hidden], dtype=self._tf_dtype)
        dvb_init = tf.zeros([self._n_visible], dtype=self._tf_dtype)
        dhb_init = tf.zeros([self._n_hidden], dtype=self._tf_dtype)
        self._dW = tf.Variable(dW_init, name='dW')
        self._dvb = tf.Variable(dvb_init, name='dvb')
        self._dhb = tf.Variable(dhb_init, name='dhb')

        # can have multiple learning rates and momentums, i think different one for each gibbs step
        # maybe for DBN or PCD
        # if you use multiple, placeholders are used so it can change values, only use one at a time
        #learning_rate=0.01
        momentum=0.9
        self._learning_rate = tf.constant(learning_rate, dtype=self._tf_dtype, name='hb_init')
        self._momentum = tf.constant(momentum, dtype=self._tf_dtype, name='hb_init')
        
        # used in sparsity penalty
        self._q_means = tf.Variable(tf.zeros([self._n_hidden], dtype=self._tf_dtype), name='q_means')
        self.sparsity_damping = .9
        self.sparsity_target = .1
        self.sparsity_cost = 0
        self._sparsity_target = tf.constant(self.sparsity_target, dtype=self._tf_dtype, name='sparsity_target')
        self._sparsity_cost = tf.constant(self.sparsity_cost, dtype=self._tf_dtype, name='sparsity_cost')
        self._sparsity_damping = tf.constant(self.sparsity_damping, dtype=self._tf_dtype, name='sparsity_damping')


    def _propup(self, v):
        t = tf.matmul(v, self._W)
        return t

    def _propdown(self, h):
        t = tf.matmul(a=h, b=self._W, transpose_b=True)
        return t

    def _means_h_given_v(self, v):
        """Compute means E(h|v)."""
        # propup multiplies v times W to get hidden values, x will be n_hidden dimensional - 3
        x  = self._propup_multiplier * self._propup(v) # multiplier is 1
        hb = self._propup_multiplier * self._hb # _hb is n_hidden dim - 3 dimensional
        h_means = self._h_layer.activation(x=x, b=hb) # sigmoid of x + hb
        return h_means

    def _sample_h_given_v(self, h_means):
        """Sample from P(h|v)."""        
        h_samples = self._h_layer.sample(means=h_means)
        return h_samples

    def _means_v_given_h(self, h):
        """Compute means E(v|h)."""
        x  = self._propdown_multiplier * self._propdown(h) # multiply h by W to get v
        vb = self._propdown_multiplier * self._vb
        # visible layer may be Gaussian rather than Bernoulli, activation just adds x and vb in that case, sigma is 1
        v_means = self._v_layer.activation(x=x, b=vb)
        return v_means

    def _sample_v_given_h(self, v_means):
        """Sample from P(v|h)."""
        v_samples = self._v_layer.sample(means=v_means)
        return v_samples

    def _make_gibbs_step(self, h_states):
        """Compute one Gibbs step."""
        # have already gone down chain to get h states, now get v states from that and get another set of h states
        v_states = v_means = self._means_v_given_h(h_states)
        if self.sample_v_states:
            v_states = self._sample_v_given_h(v_means)

        h_states = h_means = self._means_h_given_v(v_states)
        if self.sample_h_states:
            h_states = self._sample_h_given_v(h_means)

        return v_states, v_means, h_states, h_means

    def _make_gibbs_chain_fixed(self, h_states):
        v_states = v_means = h_means = None
        for _ in range(self.n_gibbs_steps[0]): # 1
            v_states, v_means, h_states, h_means = self._make_gibbs_step(h_states)
        return v_states, v_means, h_states, h_means

    def _make_gibbs_chain(self, *args, **kwargs):
        # use faster implementation (w/o while loop) when
        # number of Gibbs steps is fixed
        if len(self.n_gibbs_steps) == 1:
            return self._make_gibbs_chain_fixed(*args, **kwargs)
        else:
            return self._make_gibbs_chain_variable(*args, **kwargs)

    def sample_and_update_batch(x_batch):
        if self.dropout > 0:
            x_batch = tf.nn.dropout(x_batch, keep_prob=self.dropout)
        # multiplies batch by weights to get hidden, then adds hidden bias to it and takes sigmoid for hidden bernoulli layer
        h0_means = self._means_h_given_v(x_batch)
        # for hidden bernoulli, samples from bernoulli distribution using h0_means as probability, samples will be 0 or 1
        h0_samples = self._sample_h_given_v(h0_means)
        # set to True
        h_states = h0_samples if self.sample_h_states else h0_means
        
        v_states, v_means, _, h_means = self._make_gibbs_chain(h_states)

        # transform op?
        
        # N will be smaller in last batch
        N = tf.cast(tf.shape(x_batch)[0], dtype=self._tf_dtype)
        # compute gradients
        dW_positive = tf.matmul(x_batch, h0_means, transpose_a=True) # h0_means hidden values from initial visible state
        dW_negative = tf.matmul(v_states, h_means, transpose_a=True) # h means and v states values after full gibbs chain
        dW = (dW_positive - dW_negative) / N - self._l2 * self._W
        dvb = tf.reduce_mean(x_batch - v_states, axis=0) # == sum / N
        dhb = tf.reduce_mean(h0_means - h_means, axis=0) # == sum / N

        # optional sparsity penalty, sparsity cost 0 for now so this has no effect
        if self.sparsity_cost > 0:
            q_means = tf.reduce_sum(h_means, axis=0) # reduces to 1 row
            q_update = self._q_means.assign(self._sparsity_damping * self._q_means + \
                                            (1 - self._sparsity_damping) * q_means)
            sparsity_penalty = self._sparsity_cost * (q_update - self._sparsity_target)
            dhb -= sparsity_penalty
            dW  -= sparsity_penalty
        
        # update parameters
        dW_update = self._dW.assign(self._learning_rate * (self._momentum * self._dW + dW))
        W_update = self._W.assign_add(dW_update)

        dvb_update = self._dvb.assign(self._learning_rate * (self._momentum * self._dvb + dvb))
        vb_update = self._vb.assign_add(dvb_update)

        dhb_update = self._dhb.assign(self._learning_rate * (self._momentum * self._dhb + dhb))
        hb_update = self._hb.assign_add(dhb_update)

        # compute metrics
        l2_loss = self._l2 * tf.nn.l2_loss(self._W)
        msre = tf.reduce_mean(tf.square(x_batch - v_means))
        fe = self._free_energy(x_batch)

    def chunks(self, l, n):
        """ create chunks/batches from input data """
        return [l[i:i + n] for i in range(0, len(l), n)]
    
    def train_epoch(tensors):
        tensors = self.chunks(tensors,self.batch_size)
        num_samples = len(tensors)
        # augment last batch in case missing some data
        if len(tensors[num_samples-1]) != self.batch_size:
            diff = self.batch_size - len(tensors[num_samples-1])
            tensors[num_samples-1].extend(tensors[0][:diff])

        for batch in tensors:
            sample_and_update(batch)

    def fit(X):
        for _ in self.epochs:
            train_epoch(X)



class RBM_Impl(RBM):
    def __init__(self, learning_rate=1e-3, sigma=1.):
        self.sigma = sigma
        super(GaussianRBM, self).__init__(v_layer_cls=GaussianLayer,
                                          sigma=sigma,
                                          h_layer_cls=BernoulliLayer,
                                          learning_rate=learning_rate,
                                          epochs=epochs, batch_size=batch_size)

    def _free_energy(self, v):
        T1 = tf.divide(tf.reshape(self._vb, [1, self.n_visible]), self._sigma)
        T2 = tf.square(tf.subtract(v, T1))
        T3 = 0.5 * tf.reduce_sum(T2, axis=1)
        T4 = -tf.reduce_sum(tf.nn.softplus(self._propup(v) + self._hb), axis=1)
        fe = tf.reduce_mean(T3 + T4, axis=0)
        return fe

    '''
    def _make_placeholders(self):
        super(GaussianRBM, self)._make_placeholders()
        with tf.name_scope('input_data'):
            # divide by resp. sigmas before any operation
            self._sigma = tf.Variable(self._sigma_tmp, dtype=self._tf_dtype, name='sigma')
            self._sigma = tf.reshape(self._sigma, [1, self.n_visible])
            self._X_batch = tf.divide(self._X_batch, self._sigma)
    '''

    