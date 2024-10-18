import torch
from sklearn.model_selection import train_test_split

from src.settings import RANDOM_SEED, DEVICE


class Splitter:
    def __init__(self, x, y, idx_num, idx_cat, idx_periodic, idx_non_periodic, test_size=0.1, val_size=0.1):
        """
        Splitter class to handle data splitting and feature separation.

        :param x: Feature matrix as a numpy array or pandas DataFrame.
        :param y: Labels as a numpy array or pandas Series.
        :param idx_num: List of indices for numerical features.
        :param idx_cat: List of indices for categorical features.
        :param idx_periodic: List of indices for periodic numerical features.
        :param idx_non_periodic: List of indices for non-periodic numerical features.
        :param test_size: Proportion of the dataset to include in the test split.
        :param val_size: Proportion of the training data to include in the validation split.
        """
        self.x = x
        self.y = y
        self.idx_num = idx_num
        self.idx_cat = idx_cat
        self.idx_periodic = idx_periodic
        self.idx_non_periodic = idx_non_periodic
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = RANDOM_SEED
        self.device = DEVICE

    def split(self):
        # Split into training and test sets
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            self.x, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # Further split training data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=self.val_size, random_state=self.random_state
        )

        # Split features into numerical and categorical
        x_train_num, x_train_cat = x_train[:, self.idx_num], x_train[:, self.idx_cat]
        x_val_num, x_val_cat = x_val[:, self.idx_num], x_val[:, self.idx_cat]
        x_test_num, x_test_cat = x_test[:, self.idx_num], x_test[:, self.idx_cat]

        # Split numerical features into periodic and non-periodic
        x_train_num_p = x_train_num[:, self.idx_periodic]
        x_train_num_np = x_train_num[:, self.idx_non_periodic]
        x_val_num_p = x_val_num[:, self.idx_periodic]
        x_val_num_np = x_val_num[:, self.idx_non_periodic]
        x_test_num_p = x_test_num[:, self.idx_periodic]
        x_test_num_np = x_test_num[:, self.idx_non_periodic]

        # Convert data to tensors
        x_train_num_p_tsr = torch.tensor(x_train_num_p, dtype=torch.float32).to(self.device)
        x_train_num_np_tsr = torch.tensor(x_train_num_np, dtype=torch.float32).to(self.device)
        x_val_num_p_tsr = torch.tensor(x_val_num_p, dtype=torch.float32).to(self.device)
        x_val_num_np_tsr = torch.tensor(x_val_num_np, dtype=torch.float32).to(self.device)
        x_test_num_p_tsr = torch.tensor(x_test_num_p, dtype=torch.float32).to(self.device)
        x_test_num_np_tsr = torch.tensor(x_test_num_np, dtype=torch.float32).to(self.device)

        x_train_cat_tsr = torch.tensor(x_train_cat, dtype=torch.float32).to(self.device)
        x_val_cat_tsr = torch.tensor(x_val_cat, dtype=torch.float32).to(self.device)
        x_test_cat_tsr = torch.tensor(x_test_cat, dtype=torch.float32).to(self.device)

        y_train_tsr = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_val_tsr = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_test_tsr = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(self.device)

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
            'test': {
                'x_num_p': x_test_num_p_tsr,
                'x_num_np': x_test_num_np_tsr,
                'x_cat': x_test_cat_tsr,
                'y': y_test_tsr
            },
            'input_sizes': {
                'num_periodic_input_size': num_periodic_input_size,
                'num_non_periodic_input_size': num_non_periodic_input_size,
                'cat_input_size': cat_input_size
            }
        }
