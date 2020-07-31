from model.layers import *
import tensorflow as tf


class BaseModel(object):
    def __init__(self):
        self.hyper = None
        self.model = None
        self.custom_objects = None

    def compile(self, optimizer='adam', loss='mse', metric=None, lr=0.0005, clipnorm=1.0):
        if optimizer == 'sgd':
            op = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, clipnorm=clipnorm)
        elif optimizer == 'adam':
            op = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
        else:
            raise ValueError('{} optimizer not supported'.format(optimizer))

        self.model.compile(optimizer=op, loss=loss, metrics=metric)
        self.model.summary()
        self.hyper['optimizer'] = optimizer
        self.hyper['loss'] = loss
        self.hyper['clipnorm'] = clipnorm

    def load_weights(self, filepath, by_name=False):
        self.model.load_weights(filepath, by_name)
        self.hyper['weights'] = filepath


class InteractionNetC(BaseModel):
    def __init__(self, units_embed=32, units_conv=128, units_fc=128, num_fc_layers=2,
                 pooling='sum', dropout=0.5, regularizer=0.0, activation='relu', target=1, activation_out='linear',
                 num_atoms=0, num_features=0, num_conv_layers_intra=2, **kwargs):
        super(InteractionNetC, self).__init__()

        self.hyper = {'model': 'InteractionNetC',
                      'units_embed': units_embed,
                      'units_conv': units_conv,
                      'units_fc': units_fc,
                      'num_conv_layers_intra': num_conv_layers_intra,
                      'num_conv_layers_inter': 0,
                      'num_fc_layers': num_fc_layers,
                      'pooling': pooling,
                      'dropout': dropout,
                      'regularizer': regularizer,
                      'activation': activation,
                      'activation_out': activation_out,
                      'target': target,
                      **kwargs}

        # Input layers
        x_input = tf.keras.layers.Input(name='atom_feature_input', shape=(num_atoms, num_features))
        a_intra = tf.keras.layers.Input(name='atom_adjacency_intra_input', shape=(num_atoms, num_atoms))

        # Graph embedding layers
        h = NodeEmbedding(self.hyper['units_embed'],
                          kernel_regularizer=self.hyper['regularizer'],
                          activation=self.hyper['activation'])(x_input)
        h = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(self.hyper['dropout']))(h)
        h = NodeEmbedding(self.hyper['units_embed'],
                          kernel_regularizer=self.hyper['regularizer'],
                          activation=self.hyper['activation'])(h)

        # Normalize adjacency
        a_intra_n = Normalize()(a_intra)

        # Graph convolution
        for i in range(self.hyper['num_conv_layers_intra']):
            _h = GraphConvolution(self.hyper['units_conv'])([h, a_intra_n])
            _h = tf.keras.layers.Activation(self.hyper['activation'])(_h)
            h = tf.keras.layers.Add()([h, _h])

        # Global pooling
        h = GlobalPooling(pooling=self.hyper['pooling'])(h)

        # Fully connected layers
        for i in range(self.hyper['num_fc_layers']):
            h = tf.keras.layers.Dense(self.hyper['units_fc'],
                                      kernel_regularizer=tf.keras.regularizers.l2(self.hyper['regularizer']))(h)
            h = tf.keras.layers.Dropout(self.hyper['dropout'])(h)
            h = tf.keras.layers.Activation(self.hyper['activation'])(h)
        h = tf.keras.layers.Dense(self.hyper['target'])(h)
        x_output = tf.keras.layers.Activation(self.hyper['activation_out'])(h)

        # Build model
        self.model = tf.keras.Model(inputs=[x_input, a_intra], outputs=x_output)
        self.custom_objects = {'NodeEmbedding': NodeEmbedding,
                               'Normalize': Normalize,
                               'GraphConvolution': GraphConvolution,
                               'GlobalPooling': GlobalPooling}


class InteractionNetNC(BaseModel):
    def __init__(self, units_embed=32, units_conv=128, units_fc=128, num_fc_layers=2,
                 pooling='sum', dropout=0.5, regularizer=0.0, activation='relu', target=1, activation_out='linear',
                 num_atoms=0, num_features=0, num_conv_layers_inter=2, **kwargs):
        super(InteractionNetNC, self).__init__()

        self.hyper = {'model': 'InteractionNetNC',
                      'units_embed': units_embed,
                      'units_conv': units_conv,
                      'units_fc': units_fc,
                      'num_conv_layers_intra': 0,
                      'num_conv_layers_inter': num_conv_layers_inter,
                      'num_fc_layers': num_fc_layers,
                      'pooling': pooling,
                      'dropout': dropout,
                      'regularizer': regularizer,
                      'activation': activation,
                      'activation_out': activation_out,
                      'target': target,
                      **kwargs}

        # Input layers
        x_input = tf.keras.layers.Input(name='atom_feature_input', shape=(num_atoms, num_features))
        a_inter = tf.keras.layers.Input(name='atom_adjacency_inter_input', shape=(num_atoms, num_atoms))

        # Graph embedding layers
        h = NodeEmbedding(self.hyper['units_embed'],
                          kernel_regularizer=self.hyper['regularizer'],
                          activation=self.hyper['activation'])(x_input)
        h = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(self.hyper['dropout']))(h)
        h = NodeEmbedding(self.hyper['units_embed'],
                          kernel_regularizer=self.hyper['regularizer'],
                          activation=self.hyper['activation'])(h)

        # Normalize adjacency
        a_inter_n = Normalize()(a_inter)

        # Graph convolution
        for i in range(self.hyper['num_conv_layers_inter']):
            _h = GraphConvolution(self.hyper['units_conv'])([h, a_inter_n])
            _h = tf.keras.layers.Activation(self.hyper['activation'])(_h)
            h = tf.keras.layers.Add()([h, _h])

        # Global pooling
        h = GlobalPooling(pooling=self.hyper['pooling'])(h)

        # Fully connected layers
        for i in range(self.hyper['num_fc_layers']):
            h = tf.keras.layers.Dense(self.hyper['units_fc'],
                                      kernel_regularizer=tf.keras.regularizers.l2(self.hyper['regularizer']))(h)
            h = tf.keras.layers.Dropout(self.hyper['dropout'])(h)
            h = tf.keras.layers.Activation(self.hyper['activation'])(h)
        h = tf.keras.layers.Dense(self.hyper['target'])(h)
        x_output = tf.keras.layers.Activation(self.hyper['activation_out'])(h)

        # Build model
        self.model = tf.keras.Model(inputs=[x_input, a_inter], outputs=x_output)
        self.custom_objects = {'NodeEmbedding': NodeEmbedding,
                               'Normalize': Normalize,
                               'GraphConvolution': GraphConvolution,
                               'GlobalPooling': GlobalPooling}


class InteractionNetCNC(BaseModel):
    def __init__(self, units_embed=32, units_conv=128, units_fc=128, num_fc_layers=2,
                 pooling='sum', dropout=0.5, regularizer=0.0, activation='relu', target=1, activation_out='linear',
                 num_atoms=0, num_features=0, num_conv_layers_intra=2, num_conv_layers_inter=2, **kwargs):
        super(InteractionNetCNC, self).__init__()

        self.hyper = {'model': 'InteractionNetCNC',
                      'units_embed': units_embed,
                      'units_conv': units_conv,
                      'units_fc': units_fc,
                      'num_conv_layers_intra': num_conv_layers_intra,
                      'num_conv_layers_inter': num_conv_layers_inter,
                      'num_fc_layers': num_fc_layers,
                      'pooling': pooling,
                      'dropout': dropout,
                      'regularizer': regularizer,
                      'activation': activation,
                      'activation_out': activation_out,
                      'target': target,
                      **kwargs}

        # Input layers
        x_input = tf.keras.layers.Input(name='atom_feature_input', shape=(num_atoms, num_features))
        a_intra = tf.keras.layers.Input(name='atom_adjacency_intra_input', shape=(num_atoms, num_atoms))
        a_inter = tf.keras.layers.Input(name='atom_adjacency_inter_input', shape=(num_atoms, num_atoms))

        # Graph embedding layers
        h = NodeEmbedding(self.hyper['units_embed'],
                          kernel_regularizer=self.hyper['regularizer'],
                          activation=self.hyper['activation'])(x_input)
        h = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(self.hyper['dropout']))(h)
        h = NodeEmbedding(self.hyper['units_embed'],
                          kernel_regularizer=self.hyper['regularizer'],
                          activation=self.hyper['activation'])(h)

        # Normalize adjacency
        a_intra_n = Normalize()(a_intra)
        a_inter_n = Normalize()(a_inter)

        # Graph convolution
        for i in range(self.hyper['num_conv_layers_intra']):
            _h = GraphConvolution(self.hyper['units_conv'])([h, a_intra_n])
            _h = tf.keras.layers.Activation(self.hyper['activation'])(_h)
            h = tf.keras.layers.Add()([h, _h])

        for i in range(self.hyper['num_conv_layers_inter']):
            _h = GraphConvolution(self.hyper['units_conv'])([h, a_inter_n])
            _h = tf.keras.layers.Activation(self.hyper['activation'])(_h)
            h = tf.keras.layers.Add()([h, _h])

        # Global pooling
        h = GlobalPooling(pooling=self.hyper['pooling'])(h)

        # Fully connected layers
        for i in range(self.hyper['num_fc_layers']):
            h = tf.keras.layers.Dense(self.hyper['units_fc'],
                                      kernel_regularizer=tf.keras.regularizers.l2(self.hyper['regularizer']))(h)
            h = tf.keras.layers.Dropout(self.hyper['dropout'])(h)
            h = tf.keras.layers.Activation(self.hyper['activation'])(h)
        h = tf.keras.layers.Dense(self.hyper['target'])(h)
        x_output = tf.keras.layers.Activation(self.hyper['activation_out'])(h)

        # Build model
        self.model = tf.keras.Model(inputs=[x_input, a_intra, a_inter], outputs=x_output)
        self.custom_objects = {'NodeEmbedding': NodeEmbedding,
                               'Normalize': Normalize,
                               'GraphConvolution': GraphConvolution,
                               'GlobalPooling': GlobalPooling}
