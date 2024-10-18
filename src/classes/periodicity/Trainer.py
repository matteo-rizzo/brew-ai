import torch
from torch.utils.data import TensorDataset, DataLoader

from src.settings import DEVICE, MODEL_PERIODICITY


class Trainer:
    def __init__(self, model, criterion, optimizer, batch_size=32, num_epochs=1000):
        """
        Trainer class to handle model training and validation.

        :param model: The model to be trained.
        :param criterion: Loss function.
        :param optimizer: Optimizer for model parameters.
        :param batch_size: Batch size for training.
        :param num_epochs: Number of epochs for training.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = DEVICE

    def train(self, train_data, val_data):
        # Unpack training and validation data
        x_train_num_p, x_train_num_np, x_train_cat, y_train = (
            train_data['x_num_p'], train_data['x_num_np'], train_data['x_cat'], train_data['y']
        )
        x_val_num_p, x_val_num_np, x_val_cat, y_val = (
            val_data['x_num_p'], val_data['x_num_np'], val_data['x_cat'], val_data['y']
        )

        # Create DataLoader for training data
        train_dataset = TensorDataset(x_train_num_p, x_train_num_np, x_train_cat, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Lists to store losses
        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(self.num_epochs):
            self.model.network.train()
            epoch_loss = 0.0
            for batch in train_loader:
                batch_x_p, batch_x_np, batch_x_cat, batch_y = batch
                self.optimizer.zero_grad()
                if MODEL_PERIODICITY.startswith("tab"):
                    outputs = self.model.predict(batch_x_p, batch_x_np, batch_x_cat)
                else:
                    outputs = self.model.predict(batch_x_p, batch_x_np)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch_x_p.size(0)
            epoch_loss /= len(train_loader.dataset)
            train_losses.append(epoch_loss)

            # Validation
            self.model.network.eval()
            with torch.no_grad():
                if MODEL_PERIODICITY.startswith("tab"):
                    val_outputs = self.model.predict(x_val_num_p, x_val_num_np, x_val_cat)
                else:
                    val_outputs = self.model.predict(x_val_num_p, x_val_num_np)
                val_loss = self.criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())

            # Print progress
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss.item():.4f}")

        return train_losses, val_losses

    def get_model(self):
        return self.model