from model.trainer import RRSSTrainer
from model.models import *
from model.dataset import Dataset
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == "__main__":
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

    # Load data
    dataset = Dataset(5, normalize=False)

    # Load model
    model = InteractionNetCNC(units_embed=256, units_conv=256, units_fc=256,
                              pooling='sum', dropout=0.5, activation='relu', target=1, activation_out='linear',
                              regularizer=0.0025, num_atoms=dataset.num_atoms, num_features=dataset.num_features,
                              num_conv_layers_intra=1, num_conv_layers_inter=1, num_fc_layers=2)

    # Train model
    trainer = RRSSTrainer(fold=20, dataset=dataset, model=model)
    trainer.fit(patience=400, lr_decay=0.75, label='')
