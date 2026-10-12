from pathlib import Path

import torch
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from sklearn.datasets import load_breast_cancer
from torch.special import logit
from sklearn.preprocessing import StandardScaler

from helper import train_val_test_split, confusion_matrix_binary, classification_metrics_binary, plot_graph
from project1.baseline import majority_baseline_predict
from project1.mlbp_classifier import MLBPClassifier

torch.manual_seed(42)

data = load_breast_cancer()

X = data.data
y = data.target

X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

torch.manual_seed(42)
model = MLBPClassifier(n_features=X_train.shape[1], n_hidden=16, activation='relu')
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.01)

train_losses = []
validation_losses = []

model.run_training_record_losses(X_train_t, y_train_t, X_val_t, y_val_t, optimizer, loss_fn, train_losses,
                                 validation_losses, epochs=300)

print(classification_metrics_binary(majority_baseline_predict(y_train), y_train))

print(classification_metrics_binary(y_val, model.run_validation(X_val_t, y_val_t, loss_fn)))
plot_graph(train_losses, validation_losses, "Training and Validation Losses Lr=0.01 ")

print("______________________________________________________________________________________")

train_losses = []
validation_losses = []

print("running models with 0.0001, 0.5 learning rates")

torch.manual_seed(42)
modelA = MLBPClassifier(n_features=X_train.shape[1], n_hidden=16, activation='relu')
optimizerA = torch.optim.SGD(modelA.parameters(), 0.0001)
modelA.run_training_record_losses(X_train_t, y_train_t, X_val_t, y_val_t, optimizerA, loss_fn, train_losses,
                                  validation_losses, epochs=300)

print("Confusion matrix for model with learning rate 0.0001")
print(classification_metrics_binary(y_val, modelA.run_validation(X_val_t, y_val_t, loss_fn)))

plot_graph(train_losses, validation_losses, "Training and Validation Losses Lr=0.0001 ")

train_losses = []
validation_losses = []

print("_____________________________________________________________________________________")

torch.manual_seed(42)
modelB = MLBPClassifier(n_features=X_train.shape[1], n_hidden=16, activation='relu')
optimizerB = torch.optim.SGD(modelB.parameters(), 0.5)
best = modelB.run_training_record_losses(
    X_train_t, y_train_t, X_val_t, y_val_t,
    optimizerB, loss_fn, train_losses, validation_losses, epochs=300,
)

print("Learning rate 0.5: last epoch", len(validation_losses))
print("Last validation loss:", validation_losses[-1])
print(classification_metrics_binary(y_val, modelB.run_validation(X_val_t, y_val_t, loss_fn)))

plot_graph(
    train_losses, validation_losses,
    "Training and Validation Losses Lr=0.5",
    best_epoch=best["epoch"],
)

# Keep loss history: loading weights does not recreate these lists.
print("______________________________________________________________________________________")

print("Loading the best model with learning rate 0.5")

modelB.load_state_dict(best["state_dict"])
print("Best epoch:", best["epoch"])
print("Best validation loss:", best["val_loss"])
print(classification_metrics_binary(y_val, modelB.run_validation(X_val_t, y_val_t, loss_fn)))

checkpoint_path = Path(__file__).with_name("best_model_lr_0_5.pt")
torch.save(best, checkpoint_path)
print("Saved checkpoint:", checkpoint_path)

# Slice copies of the history through the best epoch (inclusive).
plot_graph(
    train_losses[:best["epoch"]], validation_losses[:best["epoch"]],
    "Training and Validation Losses Lr=0.5 through best epoch",
    best_epoch=best["epoch"],
)

print("______________________________________________________________________________________")

print("Running the saved best model on test set")

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
test_model = MLBPClassifier(n_features=X_train.shape[1], n_hidden=16, activation='relu')
test_model.load_state_dict(checkpoint["state_dict"])

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)
test_predictions = test_model.run_validation(X_test_t, y_test_t, loss_fn)

print(classification_metrics_binary(y_test, test_predictions))


