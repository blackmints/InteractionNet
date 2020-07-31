import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from tensorflow.keras.models import load_model
from model.dataset import Dataset
from model.layers import *
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from analysis import Draw
from analysis.Draw.MolDrawing import DrawingOptions

# Settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


@tf.function
def lrp_dense(inputs, relevance, weight, bias, epsilon=1e-5):
    # Forward pass
    z_k = tf.linalg.matmul(inputs, weight) + bias + epsilon
    # Division
    s_k = tf.math.divide_no_nan(relevance, z_k)
    # Backward pass
    c_j = tf.linalg.matmul(s_k, tf.transpose(weight))
    # Multiplication
    r_j = tf.math.multiply_no_nan(inputs, c_j)
    return r_j


@tf.function
def lrp_pooling(inputs, relevance):
    ratio = tf.math.divide_no_nan(inputs, tf.math.reduce_sum(inputs, axis=1, keepdims=True))
    r = tf.math.multiply_no_nan(tf.expand_dims(relevance, 1), ratio)
    return r


@tf.function
def lrp_add(inputs, relevance):
    input_1, input_2 = inputs
    s = input_1 + input_2
    r_1 = tf.math.multiply_no_nan(relevance, tf.math.divide_no_nan(input_1, s))
    r_2 = tf.math.multiply_no_nan(relevance, tf.math.divide_no_nan(input_2, s))
    return r_1, r_2


@tf.function
def lrp_gcn_epsilon(inputs, relevance, adjacency, weight, epsilon=1e-5):
    # GCN XW
    ax = tf.linalg.einsum('aij,ajk->aik', adjacency, inputs)
    z = tf.linalg.einsum('aij,jk->aik', ax, weight) + epsilon
    s = tf.math.divide_no_nan(relevance, z)
    c = tf.linalg.einsum('aij,jk->aik', s, tf.transpose(weight))
    r = tf.math.multiply_no_nan(ax, c)

    # GCN AX: X'^T = (AX)^T = X^T A^T
    inputs = tf.transpose(inputs, perm=[0, 2, 1])
    adjacency = tf.transpose(adjacency, perm=[0, 2, 1])
    r = tf.transpose(r, perm=[0, 2, 1])
    z = tf.linalg.einsum('aij,ajk->aik', inputs, adjacency) + epsilon
    s = tf.math.divide_no_nan(r, z)
    c = tf.linalg.einsum('aij,ajk->aik', s, tf.transpose(adjacency, perm=[0, 2, 1]))
    r = tf.math.multiply_no_nan(inputs, c)
    r = tf.transpose(r, perm=[0, 2, 1])
    return r


@tf.function
def lrp_gcn_gamma(inputs, relevance, adjacency, weight, gamma=0.1):
    # GCN XW
    weight_plus = tf.maximum(weight, tf.zeros_like(weight))
    weight = weight + gamma * weight_plus
    ax = tf.linalg.einsum('aij,ajk->aik', adjacency, inputs)
    z = tf.linalg.einsum('aij,jk->aik', ax, weight)
    s = tf.math.divide_no_nan(relevance, z)
    c = tf.linalg.einsum('aij,jk->aik', s, tf.transpose(weight))
    r = tf.math.multiply_no_nan(ax, c)

    # GCN AX: X'^T = (AX)^T = X^T A^T
    inputs = tf.transpose(inputs, perm=[0, 2, 1])
    adjacency = tf.transpose(adjacency, perm=[0, 2, 1]) * (1 + gamma)
    r = tf.transpose(r, perm=[0, 2, 1])
    z = tf.linalg.einsum('aij,ajk->aik', inputs, adjacency)
    s = tf.math.divide_no_nan(r, z)
    c = tf.linalg.einsum('aij,ajk->aik', s, tf.transpose(adjacency, perm=[0, 2, 1]))
    r = tf.math.multiply_no_nan(inputs, c)
    r = tf.transpose(r, perm=[0, 2, 1])
    return r


@tf.function
def lrp_gcn_gamma_plus(inputs, relevance, adjacency, weight):
    # GCN XW
    weight = tf.maximum(weight, tf.zeros_like(weight))
    ax = tf.linalg.einsum('aij,ajk->aik', adjacency, inputs)
    z = tf.linalg.einsum('aij,jk->aik', ax, weight)
    s = tf.math.divide_no_nan(relevance, z)
    c = tf.linalg.einsum('aij,jk->aik', s, tf.transpose(weight))
    r = tf.math.multiply_no_nan(ax, c)

    # GCN AX: X'^T = (AX)^T = X^T A^T
    inputs = tf.transpose(inputs, perm=[0, 2, 1])
    adjacency = tf.transpose(adjacency, perm=[0, 2, 1])
    r = tf.transpose(r, perm=[0, 2, 1])
    z = tf.linalg.einsum('aij,ajk->aik', inputs, adjacency)
    s = tf.math.divide_no_nan(r, z)
    c = tf.linalg.einsum('aij,ajk->aik', s, tf.transpose(adjacency, perm=[0, 2, 1]))
    r = tf.math.multiply_no_nan(inputs, c)
    r = tf.transpose(r, perm=[0, 2, 1])
    return r


def perform_lrp(model, hyper, trial=0, sample=None, epsilon=0.1, gamma=0.1):
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

    # Make folder
    fig_path = "../analysis/{}".format(model)
    if not os.path.isdir(fig_path):
        os.mkdir(fig_path)
    fig_path = "../analysis/{}/{}".format(model, hyper)
    if not os.path.isdir(fig_path):
        os.mkdir(fig_path)
    fig_path = "../analysis/{}/{}/heatmap".format(model, hyper)
    if not os.path.isdir(fig_path):
        os.mkdir(fig_path)

    # Load results
    base_path = "../result/{}/{}/".format(model, hyper)
    path = base_path + 'trial_{:02d}/'.format(trial)

    # Load hyper
    with open(path + 'hyper.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            hyper = dict(row)

    # Load model
    custom_objects = {'NodeEmbedding': NodeEmbedding,
                      'GraphConvolution': GraphConvolution,
                      'Normalize': Normalize,
                      'GlobalPooling': GlobalPooling}
    model = load_model(path + 'best_model.h5', custom_objects=custom_objects)
    print([l.name for l in model.layers])

    # Load data
    data = np.load(path + 'data_split.npz')
    dataset = Dataset('refined', 5)
    if sample is not None:
        dataset.split_by_idx(32, data['train'], data['valid'], data['test'][sample])
    else:
        dataset.split_by_idx(32, data['train'], data['valid'], data['test'])
    data.close()

    # Predict
    true_y = dataset.test_y
    outputs = {}
    for layer_name in ['node_embedding', 'node_embedding_1', 'normalize', 'normalize_1', 'activation', 'add',
                       'activation_1', 'add_1', 'global_pooling', 'activation_2', 'activation_3', 'activation_4',
                       'atom_feature_input']:
        sub_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        outputs[layer_name] = sub_model.predict(dataset.test, steps=dataset.test_step, verbose=0)[:len(true_y)]

    # Output layer: LRP-0
    # print('Calculating Dense_2...')
    relevance = lrp_dense(outputs['activation_3'],
                          outputs['activation_4'],
                          model.get_layer('dense_2').get_weights()[0],
                          model.get_layer('dense_2').get_weights()[1],
                          epsilon=0)

    # Dense layer: LRP-e
    # print('Calculating Dense_1...')
    relevance = lrp_dense(outputs['activation_2'],
                          relevance,
                          model.get_layer('dense_1').get_weights()[0],
                          model.get_layer('dense_1').get_weights()[1],
                          epsilon=epsilon)

    # Dense layer: LRP-e
    # print('Calculating Dense_0...')
    relevance = lrp_dense(outputs['global_pooling'],
                          relevance,
                          model.get_layer('dense').get_weights()[0],
                          model.get_layer('dense').get_weights()[1],
                          epsilon=epsilon)

    # Pooling layer
    # print('Calculating Pooling...')
    relevance = lrp_pooling(outputs['activation_1'],
                            relevance)

    # Add layer
    # print('Calculating Add_1...')
    relevance_1, relevance_2 = lrp_add([outputs['add'], outputs['activation_1']],
                                       relevance)

    # GCN layer: LRP-g
    # print('Calculating GCN_1...')
    relevance = lrp_gcn_gamma(outputs['add'],
                              relevance_2,
                              outputs['normalize_1'],
                              model.get_layer('graph_convolution_1').get_weights()[0],
                              gamma=gamma) + relevance_1

    # Add layer
    # print('Calculating Add_0...')
    relevance_1, relevance_2 = lrp_add([outputs['graph_embedding_1'], outputs['activation']],
                                       relevance)

    # GCN layer: LRP-g
    # print('Calculating GCN_0...')
    relevance = lrp_gcn_gamma(outputs['graph_embedding_1'],
                              relevance_2,
                              outputs['normalize'],
                              model.get_layer('graph_convolution').get_weights()[0],
                              gamma=gamma) + relevance_1

    # Embedding layer : LRP-e
    # print('Calculating Embedding_1...')
    relevance = lrp_dense(outputs['graph_embedding'],
                          relevance,
                          model.get_layer('graph_embedding_1').get_weights()[0],
                          model.get_layer('graph_embedding_1').get_weights()[1],
                          epsilon=epsilon)

    # Embedding layer : LRP-e
    # print('Calculating Embedding_0...')
    relevance = lrp_dense(outputs['atom_feature_input'],
                          relevance,
                          model.get_layer('graph_embedding').get_weights()[0],
                          model.get_layer('graph_embedding').get_weights()[1],
                          epsilon=epsilon)

    relevance = tf.math.reduce_sum(relevance, axis=-1).numpy()
    relevance = np.divide(relevance, np.expand_dims(true_y, -1))

    # Preset
    DrawingOptions.bondLineWidth = 1.5
    DrawingOptions.elemDict = {}
    DrawingOptions.dotsPerAngstrom = 20
    DrawingOptions.atomLabelFontSize = 4
    DrawingOptions.atomLabelMinFontSize = 4
    DrawingOptions.dblBondOffset = 0.3
    DrawingOptions.wedgeDashedBonds = False

    # Load data
    dataframe = pd.read_pickle('../data/5A.pkl')
    if sample is not None:
        test_set = np.load(path + 'data_split.npz')['test'][sample]
    else:
        test_set = np.load(path + 'data_split.npz')['test']

    # Draw images for test molecules
    colormap = cm.get_cmap('seismic')
    for idx, test_idx in enumerate(test_set):
        print('Drawing figure for {}/{}'.format(idx, len(test_set)))
        pdb_code = dataframe.iloc[test_idx]['code']
        error = np.absolute(dataframe.iloc[test_idx]['output'] - outputs['activation_4'][idx])[0]
        if error > 0.2: continue

        for mol_ligand, mol_pocket in zip(
                Chem.SDMolSupplier('../data/refined-set/{}/{}_ligand.sdf'.format(pdb_code, pdb_code)),
                Chem.SDMolSupplier('../data/refined-set/{}/{}_pocket.sdf'.format(pdb_code, pdb_code))):

            # Crop atoms
            mol = Chem.CombineMols(mol_ligand, mol_pocket)
            distance = np.array(rdmolops.Get3DDistanceMatrix(mol))
            cropped_idx = np.argwhere(np.min(distance[:, :mol_ligand.GetNumAtoms()], axis=1) <= 5).flatten()
            unpadded_relevance = np.zeros((mol.GetNumAtoms(),))
            np.put(unpadded_relevance, cropped_idx, relevance[idx])
            scale = max(max(unpadded_relevance), math.fabs(min(unpadded_relevance))) * 3

            # Separate fragments in Combined Mol
            idxs_frag = rdmolops.GetMolFrags(mol)
            mols_frag = rdmolops.GetMolFrags(mol, asMols=True)

            # Draw fragment and interaction
            for i, (mol_frag, idx_frag) in enumerate(zip(mols_frag[1:], idxs_frag[1:])):
                # Ignore water
                if mol_frag.GetNumAtoms() == 1:
                    continue

                # Generate 2D image
                mol_combined = Chem.CombineMols(mols_frag[0], mol_frag)
                AllChem.Compute2DCoords(mol_combined)
                fig = Draw.MolToMPL(mol_combined, coordScale=1)
                fig.axes[0].set_axis_off()

                # Draw line between close atoms (5A)
                flag = False
                for j in range(mol_ligand.GetNumAtoms()):
                    for k in idx_frag:
                        if distance[j, k] <= 5:
                            # Draw connection
                            coord_li = mol_combined._atomPs[j]
                            coord_po = mol_combined._atomPs[idx_frag.index(k) + mols_frag[0].GetNumAtoms()]
                            x, y = np.array([[coord_li[0], coord_po[0]], [coord_li[1], coord_po[1]]])
                            line = Line2D(x, y, color='b', linewidth=1, alpha=0.3)
                            fig.axes[0].add_line(line)
                            flag = True

                # Draw heatmap for atoms
                for j in range(mol_combined.GetNumAtoms()):
                    relevance_li = unpadded_relevance[j]
                    relevance_li = relevance_li / scale + 0.5
                    highlight = plt.Circle((mol_combined._atomPs[j][0],
                                            mol_combined._atomPs[j][1]),
                                           0.035 * math.fabs(unpadded_relevance[j] / scale) + 0.008,
                                           color=colormap(relevance_li), alpha=0.8, zorder=0)
                    fig.axes[0].add_artist(highlight)

                # Save
                if flag:
                    fig_name = fig_path + '/{}_lrp_{}_{}_{}.png'.format(trial, test_idx, pdb_code, i)
                    fig.savefig(fig_name, bbox_inches='tight')
                plt.close(fig)


if __name__ == "__main__":
    perform_lrp('InteractionNetCNC', '__YOUR_LOG_PATH__', trial=0, epsilon=0.25, gamma=1)
