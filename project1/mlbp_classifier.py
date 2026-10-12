from copy import deepcopy

import torch
import numpy as np

from torch import nn


class MLBPClassifier(nn.Module):
    def __init__(self, n_features, n_hidden, activation='relu'):
        super().__init__()
        activations = {
            'linear': nn.Identity,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
        }
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            activations[activation](),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, X):
        return self.net(X).squeeze(-1)

    def run_training_record_losses(
            self, X_train,
            y_train, X_val,
            y_val, optimizer,
            loss_fn, train_losses,
            val_losses,
            epochs=300):

        if epochs < 1:
            raise ValueError("epochs must be at least 1")

        best_val_loss = float("inf")
        best_epoch = None
        best_state = None

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            logits = self(X_train)
            loss = loss_fn(logits, y_train)

            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                train_loss = loss_fn(self(X_train), y_train).item()
                val_loss = loss_fn(self(X_val), y_val).item()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = deepcopy(self.state_dict())

        return {
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "state_dict": best_state,
        }

    def run_training(
            self, X_train,
            y_train, X_val,
            y_val, optimizer,
            loss_fn, train_losses,
            val_losses,
            epochs=300):

        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            logits = self(X_train)
            loss = loss_fn(logits, y_train)

            loss.backward()
            optimizer.step()

    def run_validation(self, X_val, y_val, loss_fn):
        self.eval()
        with torch.no_grad():
            val_logits = self(X_val)
            val_probabilities = torch.sigmoid(val_logits)
            val_pred_torch = (val_probabilities >= 0.5).numpy().astype(np.int64)

        return val_pred_torch




