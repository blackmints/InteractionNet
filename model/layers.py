import tensorflow as tf


class NodeEmbedding(tf.keras.layers.Layer):
    """
    A simple node embedding layer.
    """

    def __init__(self,
                 output_dim,
                 mask_zero_padding=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=0.0,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        self.supports_masking = True
        self.output_dim = output_dim
        self.mask_zero_padding = mask_zero_padding
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel = None
        self.bias = None
        self.num_atoms, self.num_features = 0, 0
        super(NodeEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_atoms = input_shape[1]
        self.num_features = input_shape[-1]
        self.kernel = self.add_weight(name='w_embed',
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight(name='b_embed',
                                        shape=(self.output_dim,))
        super(NodeEmbedding, self).build(input_shape)

    def call(self, inputs, mask=None):
        # feature = (samples, num_atoms, num_features)
        features = inputs

        # Linear combination of features
        outputs = tf.linalg.einsum('aij,jk->aik', features, self.kernel)
        if self.use_bias:
            outputs += self.bias
        outputs = tf.reshape(outputs, [-1, self.num_atoms, self.output_dim])

        # Activation
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'mask_zero_padding': self.mask_zero_padding,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': self.kernel_regularizer.l2,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        }
        base_config = super(NodeEmbedding, self).get_config()
        return {**base_config, **config}

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero_padding:
            return mask
        return tf.math.not_equal(tf.reduce_sum(inputs, axis=-1), 0)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim


class GraphConvolution(tf.keras.layers.Layer):
    """
    A simple graph convolution layer based on Kipf et al. https://arxiv.org/abs/1609.02907
    """

    def __init__(self,
                 output_dim,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activation=None,
                 **kwargs):
        self.output_dim = output_dim
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.activation = tf.keras.activations.get(activation)
        self.kernel = None
        super(GraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='gc_w',
                                      shape=(input_shape[0][-1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer)

    def call(self, inputs, **kwargs):
        # Get input tensors
        # features = (batch, num_atoms, num_features)
        # adjacency = (batch, num_atoms, num_atoms)
        features, adjacency = inputs

        # Matrix multiplication
        outputs = tf.linalg.einsum('aij,jk->aik', features, self.kernel)
        outputs = tf.linalg.einsum('aij,ajk->aik', adjacency, outputs)

        # Activation
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.output_dim

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': tf.keras.activations.serialize(self.activation),
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        }
        base_config = super(GraphConvolution, self).get_config()
        return {**base_config, **config}


class Normalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Normalize, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Get input tensors
        # adjacency = (batch, num_atoms, num_atoms)
        adjacency = inputs

        # Normalize
        atom_mask = tf.math.not_equal(tf.reduce_sum(adjacency, axis=-1), 0)
        adjacency_hat = tf.linalg.set_diag(adjacency, tf.cast(atom_mask, dtype=tf.float32))
        degree_hat = tf.reduce_sum(adjacency_hat, axis=-1)
        degree_hat_sqrt = tf.math.sqrt(degree_hat)
        degree_hat_sqrt_inv = tf.math.reciprocal_no_nan(degree_hat_sqrt)
        degree_hat_sqrt_inv = tf.linalg.diag(degree_hat_sqrt_inv)
        adjacency_norm = tf.matmul(tf.matmul(degree_hat_sqrt_inv, adjacency_hat), degree_hat_sqrt_inv)
        return adjacency_norm

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super(Normalize, self).get_config()


class GlobalPooling(tf.keras.layers.Layer):
    """
    A simple graph global pooling layer.
    """

    def __init__(self,
                 pooling="sum",
                 **kwargs):
        self.pooling = pooling
        self.num_features = 0
        super(GlobalPooling, self).__init__(**kwargs)

    def build(self, inputs_shape):
        self.num_features = inputs_shape[-1]
        super(GlobalPooling, self).build(inputs_shape)

    def call(self, inputs, mask=None, **kwargs):
        # features = (batch, num_atoms, num_features)
        features = inputs

        # Support masking
        if mask is not None:
            mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, self.num_features])
            features = tf.ragged.boolean_mask(features, mask)

        # Integrate over atom axis
        if self.pooling == "sum":
            features = tf.reduce_sum(features, axis=1)
        elif self.pooling == "max":
            features = tf.reduce_max(features, axis=1)
        elif self.pooling == "avg":
            features = tf.reduce_mean(features, axis=1)
        return features

    def compute_output_shape(self, inputs_shape):
        return inputs_shape[0], inputs_shape[-1]

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'pooling': self.pooling
        }
        base_config = super(GlobalPooling, self).get_config()
        return {**base_config, **config}
