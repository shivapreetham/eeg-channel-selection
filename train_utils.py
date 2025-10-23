"""
Training utilities for EEG-ARNN
Includes data loading, training loops, evaluation metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mne
from pathlib import Path
from tqdm.auto import tqdm


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""
    def __init__(self, data, labels):
        """
        Parameters
        ----------
        data : np.ndarray
            EEG data of shape (n_trials, n_channels, n_timepoints)
        labels : np.ndarray
            Labels of shape (n_trials,)
        """
        self.data = torch.FloatTensor(data).unsqueeze(1)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_preprocessed_data(fif_path, tmin=-1.0, tmax=5.0, baseline=(-0.5, 0)):
    """
    Load preprocessed FIF file and extract epochs

    Parameters
    ----------
    fif_path : Path
        Path to preprocessed FIF file
    tmin : float
        Start time of epoch
    tmax : float
        End time of epoch
    baseline : tuple
        Baseline correction window

    Returns
    -------
    data : np.ndarray
        Epoched data (n_epochs, n_channels, n_timepoints)
    labels : np.ndarray
        Event labels (n_epochs,)
    """
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose='ERROR')

    events = mne.find_events(raw, verbose='ERROR')

    if len(events) == 0:
        return None, None

    event_ids = {f'T{i}': i for i in np.unique(events[:, 2])}

    epochs = mne.Epochs(
        raw, events,
        event_id=event_ids,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose='ERROR'
    )

    data = epochs.get_data()
    labels = epochs.events[:, 2]

    return data, labels


def filter_classes(data, labels, selected_classes):
    """
    Filter data to include only selected classes

    Parameters
    ----------
    data : np.ndarray
        EEG data
    labels : np.ndarray
        Labels
    selected_classes : list
        List of class IDs to keep

    Returns
    -------
    filtered_data : np.ndarray
    filtered_labels : np.ndarray
        Labels remapped to 0, 1, 2, ...
    """
    mask = np.isin(labels, selected_classes)
    filtered_data = data[mask]
    filtered_labels = labels[mask]

    label_mapping = {old: new for new, old in enumerate(selected_classes)}
    filtered_labels = np.array([label_mapping[label] for label in filtered_labels])

    return filtered_data, filtered_labels


def normalize_data(data):
    """
    Z-score normalization per channel

    Parameters
    ----------
    data : np.ndarray
        Shape (n_trials, n_channels, n_timepoints)

    Returns
    -------
    normalized_data : np.ndarray
        Z-score normalized data
    """
    mean = data.mean(axis=(0, 2), keepdims=True)
    std = data.std(axis=(0, 2), keepdims=True) + 1e-8
    return (data - mean) / std


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels


def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, patience=15):
    """
    Complete training loop with early stopping

    Parameters
    ----------
    model : nn.Module
        EEG-ARNN model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    device : torch.device
        Device to train on
    epochs : int
        Maximum number of epochs
    lr : float
        Learning rate
    patience : int
        Early stopping patience

    Returns
    -------
    history : dict
        Training history
    best_model_state : dict
        State dict of best model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_model_state)

    return history, best_model_state


def cross_validate_subject(data, labels, num_channels, num_timepoints, num_classes,
                          device, n_splits=3, epochs=100, lr=0.001):
    """
    Perform k-fold cross-validation for a single subject

    Parameters
    ----------
    data : np.ndarray
        EEG data (n_trials, n_channels, n_timepoints)
    labels : np.ndarray
        Labels (n_trials,)
    num_channels : int
        Number of EEG channels
    num_timepoints : int
        Number of time points
    num_classes : int
        Number of classes
    device : torch.device
        Device to train on
    n_splits : int
        Number of CV folds
    epochs : int
        Max epochs per fold
    lr : float
        Learning rate

    Returns
    -------
    results : dict
        Cross-validation results
    """
    from models import EEGARNN

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    adjacency_matrices = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        X_train = normalize_data(X_train)
        X_val = normalize_data(X_val)

        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = EEGARNN(
            num_channels=num_channels,
            num_timepoints=num_timepoints,
            num_classes=num_classes,
            hidden_dim=40
        ).to(device)

        history, best_state = train_model(
            model, train_loader, val_loader, device,
            epochs=epochs, lr=lr, patience=15
        )

        model.load_state_dict(best_state)
        _, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, nn.CrossEntropyLoss(), device
        )

        adj_matrix = model.get_final_adjacency_matrix()
        adjacency_matrices.append(adj_matrix)

        fold_results.append({
            'fold': fold,
            'val_acc': val_acc,
            'history': history,
            'preds': val_preds,
            'labels': val_labels
        })

    avg_adjacency = np.mean(adjacency_matrices, axis=0)

    return {
        'fold_results': fold_results,
        'avg_accuracy': np.mean([r['val_acc'] for r in fold_results]),
        'std_accuracy': np.std([r['val_acc'] for r in fold_results]),
        'adjacency_matrix': avg_adjacency
    }
