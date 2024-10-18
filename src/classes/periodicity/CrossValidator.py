import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

from src.classes.periodicity.ModelFactory import ModelFactory
from src.classes.periodicity.Splitter import Splitter
from src.classes.periodicity.Trainer import Trainer
from src.settings import DEVICE


class CrossValidator:
    def __init__(self, model_name: str, x, y, idx_num, idx_cat, idx_periodic, idx_non_periodic, num_folds=5,
                 batch_size=32, num_epochs=100, learning_rate=1e-3):
        """
        CrossValidator class to handle k-fold cross-validation.

        :param x: Feature matrix as a numpy array or pandas DataFrame.
        :param y: Labels as a numpy array or pandas Series.
        :param idx_num: List of indices for numerical features.
        :param idx_cat: List of indices for categorical features.
        :param idx_periodic: List of indices for periodic numerical features.
        :param idx_non_periodic: List of indices for non-periodic numerical features.
        :param num_folds: Number of folds for cross-validation.
        :param batch_size: Batch size for training.
        :param num_epochs: Number of epochs for training.
        :param learning_rate: Learning rate for the optimizer.
        """
        self.model_name = model_name
        self.x = x
        self.y = y
        self.idx_num = idx_num
        self.idx_cat = idx_cat
        self.idx_periodic = idx_periodic
        self.idx_non_periodic = idx_non_periodic
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def run(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_val_index, test_index) in enumerate(kf.split(self.x)):
            print(f"\nFold {fold + 1}/{self.num_folds}")

            # Split data into training/validation and test sets
            x_train_val, x_test = self.x[train_val_index], self.x[test_index]
            y_train_val, y_test = self.y[train_val_index], self.y[test_index]

            # Create Splitter for this fold
            splitter = Splitter(x_train_val, y_train_val, self.idx_num, self.idx_cat, self.idx_periodic,
                                self.idx_non_periodic, test_size=0.1, val_size=0.1)

            split_data = splitter.split()

            # Extract data
            train_data = split_data['train']
            val_data = split_data['val']
            test_data = {
                'x_num_p': torch.tensor(x_test[:, self.idx_num][:, self.idx_periodic], dtype=torch.float32).to(DEVICE),
                'x_num_np': torch.tensor(x_test[:, self.idx_num][:, self.idx_non_periodic], dtype=torch.float32).to(
                    DEVICE),
                'x_cat': torch.tensor(x_test[:, self.idx_cat], dtype=torch.float32).to(DEVICE),
                'y': torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            }
            input_sizes = split_data['input_sizes']

            # Create the model using ModelFactory
            model = ModelFactory(
                num_periodic_input_size=input_sizes['num_periodic_input_size'],
                num_non_periodic_input_size=input_sizes['num_non_periodic_input_size'],
                cat_input_size=input_sizes['cat_input_size']
            ).get_model(model_name=self.model_name)

            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.network.parameters(), lr=self.learning_rate)

            # Create Trainer
            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs
            )

            # Train the model
            train_losses, val_losses = trainer.train(train_data, val_data)
            model = trainer.get_model()

            # Evaluate on test set
            model.network.eval()
            with torch.no_grad():
                if self.model_name.startswith("tab"):
                    test_outputs = model.predict(test_data['x_num_p'], test_data['x_num_np'], test_data['x_cat'])
                else:
                    test_outputs = model.predict(test_data['x_num_p'], test_data['x_num_np'])

                # Compute loss (MSE)
                test_loss = criterion(test_outputs, test_data['y']).item()

                # Detach the tensors and move them to CPU for metric computation
                test_outputs = test_outputs.cpu().numpy()
                y_test_actual = test_data['y'].cpu().numpy()

                # Compute additional metrics: R², MAE, RMSE
                r2 = r2_score(y_test_actual, test_outputs)
                mae = mean_absolute_error(y_test_actual, test_outputs)
                rmse = sqrt(mean_squared_error(y_test_actual, test_outputs))

                print(f"Test Loss (MSE): {test_loss:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            # Store results for this fold
            fold_results.append({
                'fold': fold + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_loss': test_loss,
                'r2': r2,
                'mae': mae,
                'rmse': rmse
            })

        # Aggregate results
        avg_test_loss = np.mean([r['test_loss'] for r in fold_results])
        avg_r2 = np.mean([r['r2'] for r in fold_results])
        avg_mae = np.mean([r['mae'] for r in fold_results])
        avg_rmse = np.mean([r['rmse'] for r in fold_results])

        print("\nCross-Validation Results:")
        print(f"Average Test Loss (MSE): {avg_test_loss:.4f}")
        print(f"Average R²: {avg_r2:.4f}")
        print(f"Average MAE: {avg_mae:.4f}")
        print(f"Average RMSE: {avg_rmse:.4f}")

        return fold_results
