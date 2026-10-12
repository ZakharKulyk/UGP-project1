import numpy as np
import matplotlib.pyplot as plt


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    n_train = int(train_ratio * len(X))
    n_val = int(val_ratio * len(X))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )

def confusion_matrix_binary(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def classification_metrics_binary(y_true, y_pred):
    cm = confusion_matrix_binary(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]

    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        **cm,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0.0

def plot_graph(train_losses, validation_losses, title, best_epoch=None):
    plt.figure()
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation Loss")

    if best_epoch is not None:
        plt.scatter(
            best_epoch, validation_losses[best_epoch - 1],
            color="red", label=f"Best epoch: {best_epoch}", zorder=3,
        )
        plt.axvline(best_epoch, color="red", linestyle="--", alpha=0.4)

    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
