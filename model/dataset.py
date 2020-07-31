from rdkit import Chem
from rdkit.Chem import rdmolops
from openbabel import pybel
import os
import csv
import math
import pandas as pd
import numpy as np
import tensorflow as tf


class Parser(object):
    def __init__(self, cutoff=5):
        self.cutoff = cutoff
        self.atom_type = {}
        self.num_atoms = 0
        self.x_ligand, self.x_pocket, self.pdb_code = [], [], []
        self.exclude = []

        # Load index
        self.dataframe = pd.DataFrame(columns=['code', 'resolution', 'year', 'logKd', 'output'])
        self.load_index()

        # Load data
        # Check whether file already exists
        if os.path.exists('../data/ligand_filtered.sdf') and \
                os.path.exists('../data/pocket_filtered.sdf'):
            print('Skipping format conversion')
            self.x_ligand = [mol for mol in Chem.SDMolSupplier('../data/ligand_filtered.sdf')]
            self.x_pocket = [mol for mol in Chem.SDMolSupplier('../data/pocket_filtered.sdf')]
        else:
            self.convert_dataset()
            self.validate_dataset()

        # Parse data
        self.parse_dataset()

        # Save data
        self.save_dataset()

    def load_index(self):
        # Load index file
        with open('../data/INDEX_refined_data.2018') as f:
            for _ in range(6): f.readline()
            index = f.readlines()[:-1]

        # Parse index
        for data in index:
            data = data.split("  ")[:5]
            code, resolution, year, logkd = data[0], str(data[1]), int(data[2]), float(data[3])
            output = data[4].replace('<', '=').replace('>', '=').replace('~', '=').split('=')[-1].split('M')[0]
            output = output.replace('m', 'e-3').replace('u', 'e-6').replace('n', 'e-9') \
                .replace('p', 'e-12').replace('f', 'e-15')
            output = -math.log10(float(output))

            df = pd.DataFrame([[code, resolution, year, logkd, output]],
                              columns=['code', 'resolution', 'year', 'logKd', 'output'])
            self.dataframe = self.dataframe.append(df, ignore_index=True, sort=True)
        print('Index loaded')

    def convert_dataset(self):
        dataset = os.listdir('../data/refined-set/')
        dataset.sort()
        for idx, pdb_code in enumerate(dataset):
            pdb_path = '../data/refined-set/{}/{}'.format(pdb_code, pdb_code)
            ligand_path = pdb_path + '_ligand'
            pocket_path = pdb_path + '_pocket'

            if os.path.exists(ligand_path + '.mol2') and os.path.exists(pocket_path + '.pdb'):
                # File format conversion
                try:
                    mol = Chem.MolFromPDBFile(pocket_path + '.pdb')
                    w = Chem.SDWriter(pocket_path + '.sdf')
                    w.write(mol)
                except:
                    print('Parse error on {}. Skipping to next molecule'.format(pdb_code))
                    continue
                for mol in pybel.readfile('mol2', ligand_path + '.mol2'):
                    mol.write(format='sdf', filename=ligand_path + '.sdf', overwrite=True)

                # Collection
                for mol_ligand, mol_pocket in zip(Chem.SDMolSupplier(ligand_path + '.sdf'),
                                                  Chem.SDMolSupplier(pocket_path + '.sdf')):
                    if mol_ligand is not None and mol_pocket is not None:
                        df = self.dataframe.loc[self.dataframe['code'] == pdb_code].values
                        if len(df) == 0:
                            continue
                        df = df[0]
                        mol_ligand.SetProp('_Name', '{}_ligand'.format(pdb_code))
                        mol_ligand.SetProp('resolution', str(df[1]))
                        mol_ligand.SetProp('year', str(df[2]))
                        mol_ligand.SetProp('target', str(df[3]))
                        mol_ligand.SetProp('target_calc', str(df[4]))

                        mol_pocket.SetProp('_Name', '{}_pocket'.format(pdb_code))
                        mol_pocket.SetProp('resolution', str(df[1]))
                        mol_pocket.SetProp('year', str(df[2]))
                        mol_pocket.SetProp('target', str(df[3]))
                        mol_pocket.SetProp('target_calc', str(df[4]))

                        self.pdb_code.append(pdb_code)
                        self.x_ligand.append(mol_ligand)
                        self.x_pocket.append(mol_pocket)
                print('Converted {}: {}/{}'.format(pdb_code, idx, len(dataset)))

        w = Chem.SDWriter('../data/ligand.sdf')
        for mol in self.x_ligand:
            w.write(mol)
        w = Chem.SDWriter('../data/pocket.sdf')
        for mol in self.x_pocket:
            w.write(mol)
        print('Finished format conversion')

    def validate_dataset(self):
        x_complex = []
        for pdb_code, ligand, pocket in zip(self.pdb_code, self.x_ligand, self.x_pocket):
            Chem.RemoveHs(ligand)
            Chem.RemoveHs(pocket)
            complx = Chem.CombineMols(ligand, pocket)
            complx.SetProp('_Name', '{}_complex'.format(pdb_code))
            x_complex.append(complx)

        # Validate
        self.exclude = []
        for mol in x_complex:
            print("Validating {}".format(mol.GetProp('_Name')))
            dist = []
            coordinates = np.array([list(mol.GetConformer().GetAtomPosition(j)) for j in range(mol.GetNumAtoms())])
            for i in range(mol.GetNumAtoms()):
                for j in range(i, mol.GetNumAtoms()):
                    if i == j: continue
                    dist.append(np.linalg.norm(coordinates[i] - coordinates[j]))

            dist = np.array(dist)
            if len(dist[dist < 1]) > 0:
                self.exclude.append(mol.GetProp('_Name').split('_')[0])

        # Collect results
        code_new, ligands_new, pockets_new = [], [], []
        for pdb_code, ligand, pocket in zip(self.pdb_code, self.x_ligand, self.x_pocket):
            if pdb_code in self.exclude:
                continue
            else:
                code_new.append(pdb_code)
                ligands_new.append(ligand)
                pockets_new.append(pocket)
        self.pdb_code, self.x_ligand, self.x_pocket = code_new, ligands_new, pockets_new

        # Save
        w = Chem.SDWriter('../data/ligand_filtered.sdf')
        for mol in self.x_ligand:
            w.write(mol)
        w = Chem.SDWriter('../data/pocket_filtered.sdf')
        for mol in self.x_pocket:
            w.write(mol)
        print("Molecules Validated")

    def parse_dataset(self):
        def _one_hot(x, allowable_set):
            return list(map(lambda s: x == s, allowable_set))

        # Get total types of atoms
        for ligand, pocket in zip(self.x_ligand, self.x_pocket):
            mol = Chem.CombineMols(ligand, pocket)
            self.num_atoms = max(self.num_atoms, mol.GetNumAtoms())
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol not in self.atom_type.keys():
                    self.atom_type[symbol] = 1
                else:
                    self.atom_type[symbol] += 1
        self.atom_type = {k: v for k, v in sorted(self.atom_type.items(), key=lambda item: item[1], reverse=True)}

        columns = ['code', 'symbol', 'atomic_num', 'degree', 'hybridization', 'implicit_valence', 'formal_charge',
                   'aromaticity', 'ring_size', 'num_hs', 'acid_base', 'h_donor_acceptor', 'adjacency_intra',
                   'adjacency_inter', 'distance', 'output']
        self.dataframe = pd.DataFrame(columns=columns)

        hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

        for ligand, pocket in zip(self.x_ligand, self.x_pocket):
            n_ligand = ligand.GetNumAtoms()
            mol = Chem.CombineMols(ligand, pocket)

            # Crop atoms
            adjacency = np.array(rdmolops.Get3DDistanceMatrix(mol))
            idx = np.argwhere(np.min(adjacency[:, :n_ligand], axis=1) <= self.cutoff).flatten().tolist()

            # Get tensors
            Chem.AssignStereochemistry(mol)
            hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
            hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
            acidic_match = sum(mol.GetSubstructMatches(acidic), ())
            basic_match = sum(mol.GetSubstructMatches(basic), ())
            ring = mol.GetRingInfo()

            m = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
            m[0] = ligand.GetProp('_Name').split('_')[0]
            for atom_idx in idx:
                atom = mol.GetAtomWithIdx(atom_idx)
                m[1].append(_one_hot(atom.GetSymbol(), self.atom_type.keys()))
                m[2].append([atom.GetAtomicNum()])
                m[3].append(_one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
                m[4].append(_one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                               Chem.rdchem.HybridizationType.SP2,
                                                               Chem.rdchem.HybridizationType.SP3,
                                                               Chem.rdchem.HybridizationType.SP3D,
                                                               Chem.rdchem.HybridizationType.SP3D2]))
                m[5].append(_one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
                m[6].append(_one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]))
                m[7].append([atom.GetIsAromatic()])
                m[8].append([ring.IsAtomInRingOfSize(atom_idx, 3),
                             ring.IsAtomInRingOfSize(atom_idx, 4),
                             ring.IsAtomInRingOfSize(atom_idx, 5),
                             ring.IsAtomInRingOfSize(atom_idx, 6),
                             ring.IsAtomInRingOfSize(atom_idx, 7),
                             ring.IsAtomInRingOfSize(atom_idx, 8)])
                m[9].append(_one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))
                m[10].append([atom_idx in acidic_match, atom_idx in basic_match])
                m[11].append([atom_idx in hydrogen_donor_match, atom_idx in hydrogen_acceptor_match])
            m[12] = np.array(rdmolops.GetAdjacencyMatrix(mol))[idx][:, idx]
            adj = np.zeros_like(m[12])
            adj[:n_ligand, n_ligand:] = 1.
            adj[n_ligand:, :n_ligand] = 1.
            m[13] = adj
            m[14] = np.array(rdmolops.Get3DDistanceMatrix(mol))[idx][:, idx]
            m[15] = float(ligand.GetProp('target_calc'))

            self.dataframe = self.dataframe.append(pd.DataFrame([m], columns=columns), ignore_index=True, sort=True)

        # Pad data
        self.num_atoms = 0
        for i in range(len(self.dataframe)):
            self.num_atoms = max(len(self.dataframe.iloc[i]['symbol']), self.num_atoms)

        for i in range(len(self.dataframe)):
            # ['acid_base', 'adjacency_inter', 'adjacency_intra', 'aromaticity', 'atomic_num', 'code', 'degree',
            #  'distance', 'formal_charge', 'h_donor_acceptor', 'hybridization', 'implicit_valence', 'num_hs',
            #  'output', 'ring_size', 'symbol']
            delta = self.num_atoms - len(self.dataframe.iat[i, 0])
            for j in [1, 2, 7]:
                self.dataframe.iat[i, j] = np.pad(self.dataframe.iat[i, j], ((0, delta), (0, delta)), 'constant',
                                                  constant_values=((0, 0), (0, 0)))
            for j in [0, 3, 6, 8, 9, 10, 11, 12, 14, 15]:
                self.dataframe.iat[i, j] = np.pad(self.dataframe.iat[i, j], ((0, delta), (0, 0)), 'constant',
                                                  constant_values=((False, False), (False, False)))

            self.dataframe.iat[i, 4] = np.pad(self.dataframe.iat[i, 4], ((0, delta), (0, 0)), 'constant',
                                              constant_values=((0, 0), (0, 0)))

    def save_dataset(self):
        self.dataframe.to_pickle('../data/' + str(self.cutoff) + 'A.pkl')
        print('DataFrame saved')


class Dataset(object):
    def __init__(self, cutoff, normalize=False):
        self.cutoff = cutoff
        self.normalize = normalize
        self.mean, self.std = 0, 1
        self.num_atoms = 0
        self.num_features = 0
        self.data_idx = None
        self.data_idx_train = None
        self.data_idx_valid = None
        self.data_idx_test = None
        self.train, self.valid, self.test = None, None, None
        self.train_x, self.valid_x, self.test_x = None, None, None
        self.train_y, self.valid_y, self.test_y = None, None, None
        self.train_a_intra, self.valid_a_intra, self.test_a_intra = None, None, None
        self.train_a_inter, self.valid_a_inter, self.test_a_inter = None, None, None
        self.train_step, self.valid_step, self.test_step = 0, 0, 0

        # Load dataframe
        self.dataframe = pd.read_pickle('../data/{}A.pkl'.format(self.cutoff))

        # Get maximum number of atoms
        self.num_atoms = len(self.dataframe.iloc[0]['symbol'])

        # Get length of features
        features = ['acid_base', 'aromaticity', 'atomic_num', 'degree', 'formal_charge', 'h_donor_acceptor',
                    'hybridization', 'implicit_valence', 'num_hs', 'ring_size', 'symbol']
        for f in features:
            self.num_features += len(self.dataframe.iloc[0][f][0])
        print('{}A Dataset: Maximum {} atoms, {} features'.format(self.cutoff, self.num_atoms, self.num_features))

        # Construct tensors
        self.x = np.zeros((len(self.dataframe), self.num_atoms, self.num_features))
        self.a_intra = np.zeros((len(self.dataframe), self.num_atoms, self.num_atoms))
        self.a_inter = np.zeros((len(self.dataframe), self.num_atoms, self.num_atoms))
        for idx in range(len(self.dataframe)):
            self.x[idx] = np.concatenate([np.array(self.dataframe.iloc[idx][k], dtype=float) for k in features], axis=1)
            self.a_intra[idx] = np.array(self.dataframe.iloc[idx]['adjacency_intra'], dtype=float)
            # Apply cutoff
            a = np.array(self.dataframe.iloc[idx]['adjacency_inter'], dtype=float)
            d = np.array(self.dataframe.iloc[idx]['distance'], dtype=float)
            self.a_inter[idx] = np.where(d < self.cutoff, a, 0)
        self.y = self.dataframe['output'].to_numpy()

        print('Tensors x: {}, y: {}, a_intra: {}, a_inter: {}'.format(self.x.shape, self.y.shape,
                                                                      self.a_intra.shape, self.a_inter.shape))
        self.hyper = {'data_path': '../data/{}A.pkl'.format(self.cutoff),
                      'normalize': self.normalize,
                      'num_atoms': self.num_atoms,
                      'num_features': self.num_features}

    def shuffle(self):
        self.data_idx = np.random.permutation(len(self.y))

    def split(self, batch=32, valid_ratio=0.1, test_ratio=0.1):
        # Split dataset
        idx_valid = int(len(self.y) * (1 - valid_ratio - test_ratio))
        idx_test = int(len(self.y) * (1 - test_ratio))
        self.data_idx_train = self.data_idx[:idx_valid]
        self.data_idx_valid = self.data_idx[idx_valid:idx_test]
        self.data_idx_test = self.data_idx[idx_test:]

        self.train_x, self.valid_x, self.test_x = self.x[self.data_idx_train], \
                                                  self.x[self.data_idx_valid], \
                                                  self.x[self.data_idx_test]
        self.train_y, self.valid_y, self.test_y = self.y[self.data_idx_train], \
                                                  self.y[self.data_idx_valid], \
                                                  self.y[self.data_idx_test]
        self.train_a_intra, self.valid_a_intra, self.test_a_intra = self.a_intra[self.data_idx_train], \
                                                                    self.a_intra[self.data_idx_valid], \
                                                                    self.a_intra[self.data_idx_test]
        self.train_a_inter, self.valid_a_inter, self.test_a_inter = self.a_inter[self.data_idx_train], \
                                                                    self.a_inter[self.data_idx_valid], \
                                                                    self.a_inter[self.data_idx_test]

        self.hyper['num_train'] = len(self.train_y)
        self.hyper['num_valid'] = len(self.valid_y)
        self.hyper['num_test'] = len(self.test_y)

        # Normalize
        if self.normalize and self.mean == 0:
            self.mean = np.mean(self.train_y)
            self.std = np.std(self.train_y)
            self.train_y = (self.train_y - self.mean) / self.std
            self.valid_y = (self.valid_y - self.mean) / self.std
            self.test_y = (self.test_y - self.mean) / self.std
            self.hyper['mean'] = self.mean
            self.hyper['std'] = self.std
        else:
            self.hyper['mean'] = 0
            self.hyper['std'] = 1

        # Build dataset
        self.hyper['batch'] = batch
        self.train = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices((self.train_x, self.train_a_intra,
                                                                              self.train_a_inter)),
                                          tf.data.Dataset.from_tensor_slices(self.train_y))).repeat().batch(batch)
        self.valid = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices((self.valid_x, self.valid_a_intra,
                                                                              self.valid_a_inter)),
                                          tf.data.Dataset.from_tensor_slices(self.valid_y))).repeat().batch(batch)
        self.test = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices((self.test_x, self.test_a_intra,
                                                                             self.test_a_inter)),
                                         tf.data.Dataset.from_tensor_slices(self.test_y))).repeat().batch(batch)

        self.train_step = math.ceil(len(self.train_y) / batch)
        self.valid_step = math.ceil(len(self.valid_y) / batch)
        self.test_step = math.ceil(len(self.test_y) / batch)

        # Prefetch dataset
        self.train = self.train.prefetch(1)
        self.valid = self.valid.prefetch(1)
        self.test = self.test.prefetch(1)

    def split_by_idx(self, batch, data_idx_train, data_idx_valid, data_idx_test):
        self.data_idx_train = data_idx_train
        self.data_idx_valid = data_idx_valid
        self.data_idx_test = data_idx_test

        self.train_x, self.valid_x, self.test_x = self.x[self.data_idx_train], \
                                                  self.x[self.data_idx_valid], \
                                                  self.x[self.data_idx_test]
        self.train_y, self.valid_y, self.test_y = self.y[self.data_idx_train], \
                                                  self.y[self.data_idx_valid], \
                                                  self.y[self.data_idx_test]
        self.train_a_intra, self.valid_a_intra, self.test_a_intra = self.a_intra[self.data_idx_train], \
                                                                    self.a_intra[self.data_idx_valid], \
                                                                    self.a_intra[self.data_idx_test]
        self.train_a_inter, self.valid_a_inter, self.test_a_inter = self.a_inter[self.data_idx_train], \
                                                                    self.a_inter[self.data_idx_valid], \
                                                                    self.a_inter[self.data_idx_test]

        self.hyper['num_train'] = len(self.train_y)
        self.hyper['num_valid'] = len(self.valid_y)
        self.hyper['num_test'] = len(self.test_y)

        # Normalize
        if self.normalize and self.mean == 0:
            self.mean = np.mean(self.train_y)
            self.std = np.std(self.train_y)
            self.train_y = (self.train_y - self.mean) / self.std
            self.valid_y = (self.valid_y - self.mean) / self.std
            self.test_y = (self.test_y - self.mean) / self.std
            self.hyper['mean'] = self.mean
            self.hyper['std'] = self.std
        else:
            self.hyper['mean'] = 0
            self.hyper['std'] = 1

        # Build dataset
        self.train = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices((self.train_x, self.train_a_intra,
                                                                              self.train_a_inter)),
                                          tf.data.Dataset.from_tensor_slices(self.train_y))).repeat().batch(batch)
        self.valid = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices((self.valid_x, self.valid_a_intra,
                                                                              self.valid_a_inter)),
                                          tf.data.Dataset.from_tensor_slices(self.valid_y))).repeat().batch(batch)
        self.test = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices((self.test_x, self.test_a_intra,
                                                                             self.test_a_inter)),
                                         tf.data.Dataset.from_tensor_slices(self.test_y))).repeat().batch(batch)

        self.train_step = math.ceil(len(self.train_y) / batch)
        self.valid_step = math.ceil(len(self.valid_y) / batch)
        self.test_step = math.ceil(len(self.test_y) / batch)

        # Prefetch dataset
        self.train = self.train.prefetch(1)
        self.valid = self.valid.prefetch(1)
        self.test = self.test.prefetch(1)

    def save(self, file_path):
        np.savez(file_path, train=self.data_idx_train, valid=self.data_idx_valid, test=self.data_idx_test)


if __name__ == "__main__":
    Parser(cutoff=5)
