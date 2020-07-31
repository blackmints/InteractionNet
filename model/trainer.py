from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, TerminateOnNaN
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import model_from_json
import numpy as np
import time
import csv


class RRSSTrainer(object):
    """
    K-Fold Repeated Random Sub-Sampling Trainer
    """

    def __init__(self, fold=20, dataset=None, model=None):
        self.dataset = dataset
        self.model = model
        self.hyper = {'fold': fold, **dataset.hyper, **model.hyper}

    def fit(self, patience=20, lr_decay=0.5, loss='mse', label=''):
        results = []
        now = time.time()
        base_path = '../result/{}/'.format(self.hyper['model'])
        log_path = base_path + "l{}_{}_{}_e{}_{}_{}/".format(self.hyper['num_conv_layers_intra'],
                                                             self.hyper['num_conv_layers_inter'],
                                                             self.hyper['num_fc_layers'],
                                                             self.hyper['units_embed'],
                                                             label,
                                                             time.strftime('%b%d_%H_%M_%S', time.localtime(now)))
        self.hyper['patience'] = patience
        self.hyper['lr_decay'] = lr_decay

        for trial in range(int(self.hyper['fold'])):
            # Make folder
            now = time.time()
            trial_path = log_path + 'trial_{:02d}/'.format(trial)

            # Reset model
            self.model.model = model_from_json(self.model.model.to_json(), custom_objects=self.model.custom_objects)
            self.model.compile(optimizer='adam', loss=loss, lr=0.00015, clipnorm=0.5,
                               metric=[MeanAbsoluteError(), RootMeanSquaredError()])
            self.hyper = {**self.hyper, **self.model.hyper}

            # Shuffle, split and normalize data
            self.dataset.shuffle()
            self.dataset.split(batch=32, valid_ratio=0.1, test_ratio=0.1)
            self.hyper = {**self.hyper, **self.dataset.hyper}

            # Define callbacks
            callbacks = [TensorBoard(log_dir=trial_path, write_graph=False, histogram_freq=0, write_images=False),
                         EarlyStopping(patience=patience, restore_best_weights=True),
                         ReduceLROnPlateau(factor=lr_decay, patience=patience // 2),
                         TerminateOnNaN()]

            # Train model
            self.model.model.fit(self.dataset.train, steps_per_epoch=self.dataset.train_step,
                                 validation_data=self.dataset.valid, validation_steps=self.dataset.valid_step,
                                 epochs=1500, callbacks=callbacks, verbose=2)

            # Save current state
            self.model.model.save_weights(trial_path + 'best_weights.h5')
            self.model.model.save(trial_path + 'best_model.h5')
            self.hyper['training_time'] = '{:.2f}'.format(time.time() - now)

            # Evaluate model
            train_loss = self.model.model.evaluate(self.dataset.train, steps=self.dataset.train_step, verbose=0)
            valid_loss = self.model.model.evaluate(self.dataset.valid, steps=self.dataset.valid_step, verbose=0)
            test_loss = self.model.model.evaluate(self.dataset.test, steps=self.dataset.test_step, verbose=0)
            results.append([train_loss[1], valid_loss[1], test_loss[1], train_loss[2], valid_loss[2], test_loss[2]])

            # Save trial results
            with open(trial_path + 'hyper.csv', 'w') as file:
                writer = csv.DictWriter(file, fieldnames=list(self.hyper.keys()))
                writer.writeheader()
                writer.writerow(self.hyper)
            with open(trial_path + 'result.csv', 'w') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(['train_mae', 'valid_mae', 'test_mae',
                                 'train_rmse', 'valid_rmse', 'test_rmse'])
                writer.writerow(np.array(results[-1]) * self.hyper['std'])
            self.dataset.save(trial_path + 'data_split.npz')
            clear_session()

        # Save cross-validated results
        header = ['train_mae', 'valid_mae', 'test_mae', 'train_rmse', 'valid_rmse', 'test_rmse']
        results = np.array(results) * self.hyper['std']
        results = [np.mean(results, axis=0), np.std(results, axis=0)]
        with open(log_path + "results.csv", "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(header)
            for r in results:
                writer.writerow(r)
        print('{}-fold cross-validation result'.format(self.hyper['fold']))
        print('RMSE {}+-{}, {}+-{}, {}+-{}'.format(results[0][3], results[1][3], results[0][4],
                                                   results[1][4], results[0][5], results[1][5]))


class LogToTrainer(object):
    """
    Trainer for reusing the same data split used in other experiment logs
    """

    def __init__(self, trial=None, dataset=None, model=None):
        self.dataset = dataset
        self.model = model
        self.trial = trial
        self.hyper = {**dataset.hyper, **model.hyper}

    def fit(self, patience=20, lr_decay=0.5, loss='mse', label='', log_copy_path=''):
        results = []
        now = time.time()
        base_path = '../result/{}/'.format(self.hyper['model'])
        log_path = base_path + "l{}_{}_{}_e{}_{}_{}/".format(self.hyper['num_conv_layers_intra'],
                                                             self.hyper['num_conv_layers_inter'],
                                                             self.hyper['num_fc_layers'],
                                                             self.hyper['units_embed'],
                                                             label,
                                                             time.strftime('%b%d_%H_%M_%S', time.localtime(now)))
        self.hyper['patience'] = patience
        self.hyper['lr_decay'] = lr_decay
        self.hyper['log_copy_path'] = log_copy_path

        for trial in self.trial:
            # Make folder
            now = time.time()
            trial_path = log_path + 'trial_{:02d}/'.format(trial)
            trial_copy_path = log_copy_path + 'trial_{:02d}/'.format(trial)

            # Reset model
            self.model.model = model_from_json(self.model.model.to_json(), custom_objects=self.model.custom_objects)
            self.model.compile(optimizer='adam', loss=loss, lr=0.00015, clipnorm=0.5,
                               metric=[MeanAbsoluteError(), RootMeanSquaredError()])
            self.hyper = {**self.hyper, **self.model.hyper}

            # Load used data
            data = np.load(trial_copy_path + 'data_split.npz')
            self.dataset.split_by_idx(32, data['train'], data['valid'], data['test'])
            self.hyper = {**self.hyper, **self.dataset.hyper}

            # Define callbacks
            callbacks = [TensorBoard(log_dir=trial_path, write_graph=False, histogram_freq=0, write_images=False),
                         EarlyStopping(patience=patience, restore_best_weights=True),
                         ReduceLROnPlateau(factor=lr_decay, patience=patience // 2),
                         TerminateOnNaN()]

            # Train model
            self.model.model.fit(self.dataset.train, steps_per_epoch=self.dataset.train_step,
                                 validation_data=self.dataset.valid, validation_steps=self.dataset.valid_step,
                                 epochs=1500, callbacks=callbacks, verbose=2)

            # Save current state
            self.model.model.save_weights(trial_path + 'best_weights.h5')
            self.model.model.save(trial_path + 'best_model.h5')
            self.hyper['training_time'] = '{:.2f}'.format(time.time() - now)

            # Evaluate model
            train_loss = self.model.model.evaluate(self.dataset.train, steps=self.dataset.train_step, verbose=0)
            valid_loss = self.model.model.evaluate(self.dataset.valid, steps=self.dataset.valid_step, verbose=0)
            test_loss = self.model.model.evaluate(self.dataset.test, steps=self.dataset.test_step, verbose=0)
            results.append([train_loss[1], valid_loss[1], test_loss[1], train_loss[2], valid_loss[2], test_loss[2]])

            # Save trial results
            with open(trial_path + 'hyper.csv', 'w') as file:
                writer = csv.DictWriter(file, fieldnames=list(self.hyper.keys()))
                writer.writeheader()
                writer.writerow(self.hyper)
            with open(trial_path + 'result.csv', 'w') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(['train_mae', 'valid_mae', 'test_mae',
                                 'train_rmse', 'valid_rmse', 'test_rmse'])
                writer.writerow(np.array(results[-1]) * self.hyper['std'])
            self.dataset.save(trial_path + 'data_split.npz')
            clear_session()

        # Save cross-validated results
        header = ['train_mae', 'valid_mae', 'test_mae', 'train_rmse', 'valid_rmse', 'test_rmse']
        results = np.array(results) * self.hyper['std']
        results = [np.mean(results, axis=0), np.std(results, axis=0)]
        with open(log_path + "results.csv", "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(header)
            for r in results:
                writer.writerow(r)
        print('{}-fold cross-validation result'.format(self.hyper['fold']))
        print('RMSE {}+-{}, {}+-{}, {}+-{}'.format(results[0][3], results[1][3], results[0][4],
                                                   results[1][4], results[0][5], results[1][5]))
