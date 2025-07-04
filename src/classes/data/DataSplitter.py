import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, DEVICE, TEST_SIZE


class DataSplitter:
    def __init__(self, x, y, idx_num, idx_cat, idx_periodic, idx_non_periodic):
        """
        DataSplitter class to handle data splitting and feature separation.

        :param x: Feature matrix as a numpy array or pandas DataFrame.
        :param y: Labels as a numpy array or pandas Series.
        :param idx_num: List of indices for numerical features.
        :param idx_cat: List of indices for categorical features.
        :param idx_periodic: List of indices for periodic numerical features.
        :param idx_non_periodic: List of indices for non-periodic numerical features.
        """
        self.x = x
        self.y = y
        self.idx_num = idx_num
        self.idx_cat = idx_cat
        self.idx_periodic = idx_periodic
        self.idx_non_periodic = idx_non_periodic
        self.test_size = TEST_SIZE
        self.random_state = RANDOM_SEED
        self.device = DEVICE

    def split(self):
        # Further split training data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            self.x, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # FIX: Use .iloc for positional indexing on pandas DataFrames.
        # .values converts the result to a clean NumPy array for subsequent slicing.
        x_train_num = x_train.iloc[:, self.idx_num].values
        x_train_cat = x_train.iloc[:, self.idx_cat].values
        x_val_num = x_val.iloc[:, self.idx_num].values
        x_val_cat = x_val.iloc[:, self.idx_cat].values

        # Split numerical features into periodic and non-periodic
        # This slicing is now correct because x_train_num and x_val_num are NumPy arrays.
        x_train_num_p = x_train_num[:, self.idx_periodic]
        x_train_num_np = x_train_num[:, self.idx_non_periodic]
        x_val_num_p = x_val_num[:, self.idx_periodic]
        x_val_num_np = x_val_num[:, self.idx_non_periodic]

        # Convert data to tensors
        x_train_num_p_tsr = torch.tensor(x_train_num_p, dtype=torch.float32).to(self.device)
        x_train_num_np_tsr = torch.tensor(x_train_num_np, dtype=torch.float32).to(self.device)
        x_val_num_p_tsr = torch.tensor(x_val_num_p, dtype=torch.float32).to(self.device)
        x_val_num_np_tsr = torch.tensor(x_val_num_np, dtype=torch.float32).to(self.device)

        # The result of .iloc[:, []] on an empty list of columns is an empty array,
        # which needs to be handled. Reshape if it's empty.
        if x_train_cat.shape[1] == 0:
            x_train_cat = x_train_cat.reshape(len(x_train_cat), 0)
            x_val_cat = x_val_cat.reshape(len(x_val_cat), 0)

        x_train_cat_tsr = torch.tensor(x_train_cat, dtype=torch.float32).to(self.device)
        x_val_cat_tsr = torch.tensor(x_val_cat, dtype=torch.float32).to(self.device)

        # Ensure y is a numpy array before creating a tensor
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
        y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val

        y_train_tsr = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_val_tsr = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Determine input sizes
        num_periodic_input_size = x_train_num_p_tsr.shape[1]
        num_non_periodic_input_size = x_train_num_np_tsr.shape[1]
        cat_input_size = x_train_cat_tsr.shape[1]

        # Return a dictionary containing all the split data and input sizes
        return {
            'train': {
                'x_num_p': x_train_num_p_tsr,
                'x_num_np': x_train_num_np_tsr,
                'x_cat': x_train_cat_tsr,
                'y': y_train_tsr
            },
            'val': {
                'x_num_p': x_val_num_p_tsr,
                'x_num_np': x_val_num_np_tsr,
                'x_cat': x_val_cat_tsr,
                'y': y_val_tsr
            },
            'input_sizes': {
                'num_periodic_input_size': num_periodic_input_size,
                'num_non_periodic_input_size': num_non_periodic_input_size,
                'cat_input_size': cat_input_size
            }
        }
