#!/usr/bin/env python3
"""
Train a neural network on Portuguese Bank Marketing data using torch_lite.

Dataset: Predict if a client will subscribe to a term deposit (binary classification).
Model: 3-layer MLP (10→64→32→1) trained from scratch with our custom library.
"""
import sys
import os
import time
import math

# Allow running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch_lite as tl
from torch_lite.data import DataLoader, train_test_split


def load_bank_marketing(csv_path, max_rows=None):
    """Load bank marketing CSV with semicolon separator.
    
    Returns X (numerical features), y (binary labels).
    """
    print(f"Loading data from {csv_path}...")
    
    # Read CSV manually (avoid pandas dependency for pure torch_lite demo)
    with open(csv_path, 'r') as f:
        header_raw = f.readline().strip().split(';')
        lines = f.readlines()
    
    # Strip quotes from header
    header = [h.strip('"').strip() for h in header_raw]
    
    if max_rows:
        # Shuffle to get a mix of positive/negative samples
        np.random.seed(42)
        indices = np.random.permutation(len(lines))[:max_rows]
        lines = [lines[i] for i in indices]
    
    # Find which columns to use
    num_cols = [
        'age', 'duration', 'campaign', 'pdays', 'previous',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
    ]
    # Map original names (with dots) to clean names (with underscores)
    col_map = {
        'emp.var.rate': 'emp_var_rate',
        'cons.price.idx': 'cons_price_idx',
        'cons.conf.idx': 'cons_conf_idx',
        'nr.employed': 'nr_employed',
    }
    
    # Build column index mapping
    col_indices = {}
    for i, col in enumerate(header):
        col_indices[col] = i
    
    # Find indices of numerical columns
    num_indices = []
    num_names = []
    for col in num_cols:
        if col in col_indices:
            num_indices.append(col_indices[col])
            num_names.append(col_map.get(col, col))
    
    print(f"  Using {len(num_indices)} numerical features: {num_names}")
    
    # Parse data
    n = len(lines)
    X = np.zeros((n, len(num_indices)), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    
    skipped = 0
    for row_idx, line in enumerate(lines):
        parts = line.strip().split(';')
        if len(parts) < len(header):
            skipped += 1
            continue
        
        try:
            # Extract features (strip quotes from values)
            for j, col_idx in enumerate(num_indices):
                val = parts[col_idx].strip('"').strip()
                if val == 'unknown' or val == 'nonexistent' or val == '':
                    val = '0'
                X[row_idx, j] = float(val)
            
            # Extract target
            target_col = col_indices.get('y', col_indices.get('subscribed', -1))
            if target_col < 0:
                raise ValueError(f"Cannot find target column 'y' in {list(col_indices.keys())}")
            y[row_idx] = 1.0 if parts[target_col].strip('"').strip().lower() == 'yes' else 0.0
        except (ValueError, IndexError) as e:
            X[row_idx, :] = 0
            y[row_idx] = 0
            skipped += 1
    
    # Remove rows with NaN/Inf
    valid = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    X = X[valid]
    y = y[valid]
    
    print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Skipped {skipped} invalid rows")
    print(f"  Positive class: {y.sum():.0f} ({y.mean()*100:.1f}%)")
    
    return X, y


def standardize(X_train, X_val, X_test):
    """Z-score normalize using training set statistics."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    return X_train, X_val, X_test, mean, std


def compute_metrics(y_true, y_pred_prob):
    """Compute accuracy, precision, recall, F1, and AUC."""
    y_true = y_true.flatten()
    y_pred_prob = y_pred_prob.flatten()
    y_pred = (y_pred_prob >= 0.5).astype(np.float32)
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # AUC (simple trapezoidal rule)
    sorted_idx = np.argsort(y_pred_prob)[::-1]
    y_sorted = y_true[sorted_idx]
    
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        tpr = np.cumsum(y_sorted) / n_pos
        fpr = np.cumsum(1 - y_sorted) / n_neg
        # Trapezoidal rule: sum((x2-x1) * (y1+y2)/2)
        auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0)
        auc = abs(auc)  # ensure positive
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def main():
    # ── Config ──
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'data',
                             'bank-additional', 'bank-additional-full.csv')
    MAX_ROWS = 2000          # Use 2000 samples (fast demo)
    BATCH_SIZE = 64
    LEARNING_RATE = 0.02
    MOMENTUM = 0.9
    EPOCHS = 80
    HIDDEN1 = 64
    HIDDEN2 = 32
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15          # validation split from train

    # ── Load data ──
    print("=" * 60)
    print(" torch_lite Demo: Bank Marketing Classification")
    print("=" * 60)
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        print("Download from: https://archive.ics.uci.edu/dataset/222/bank+marketing")
        print("Place in: notebooks/data/bank-additional/bank-additional-full.csv")
        sys.exit(1)
    
    X, y = load_bank_marketing(DATA_PATH, max_rows=MAX_ROWS)
    
    # ── Train/Val/Test split ──
    # First split: train_val / test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )
    # Second split: train / val
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42
    )
    
    print(f"\nData splits: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    
    # ── Standardize ──
    X_train, X_val, X_test, feat_mean, feat_std = standardize(X_train, X_val, X_test)
    
    # ── DataLoaders ──
    train_loader = DataLoader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
    
    # ── Build model ──
    n_features = X_train.shape[1]
    print(f"\nBuilding MLP: {n_features} → {HIDDEN1} → {HIDDEN2} → 1")
    
    model = tl.nn.Sequential(
        tl.nn.Linear(n_features, HIDDEN1),
        tl.nn.ReLU(),
        tl.nn.Linear(HIDDEN1, HIDDEN2),
        tl.nn.ReLU(),
        tl.nn.Linear(HIDDEN2, 1),
    )
    
    criterion = tl.nn.BCELoss()
    optimizer = tl.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    
    n_params = sum(p.numel for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # ── Training loop ──
    print(f"\nTraining for {EPOCHS} epochs (lr={LEARNING_RATE}, batch={BATCH_SIZE})...")
    print("-" * 60)
    
    best_val_auc = 0.0
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    t_start = time.time()
    
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            logits = model(batch.x)
            loss = criterion(logits, batch.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss.numpy().flatten()[0])
            n_batches += 1
        
        avg_train_loss = total_loss / max(n_batches, 1)
        history['train_loss'].append(avg_train_loss)
        
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_targets = []
        n_val_batches = 0
        
        for batch in val_loader:
            logits = model(batch.x)
            loss = criterion(logits, batch.y)
            val_loss += float(loss.numpy().flatten()[0])
            prob = 1.0 / (1.0 + np.exp(-logits.numpy()))
            val_probs.append(prob.flatten())
            val_targets.append(batch.y.numpy().flatten())
            n_val_batches += 1
        
        avg_val_loss = val_loss / max(n_val_batches, 1)
        history['val_loss'].append(avg_val_loss)
        
        val_probs = np.concatenate(val_probs)
        val_targets = np.concatenate(val_targets)
        val_metrics = compute_metrics(val_targets, val_probs)
        history['val_auc'].append(val_metrics['auc'])
        
        # Track best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch + 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"val_loss={avg_val_loss:.4f} | "
                  f"val_auc={val_metrics['auc']:.4f} | "
                  f"val_acc={val_metrics['accuracy']:.4f}")
    
    train_time = time.time() - t_start
    print(f"\nTraining complete in {train_time:.1f}s")
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    
    # ── Final test evaluation ──
    print("\n" + "=" * 60)
    print(" Final Test Evaluation")
    print("=" * 60)
    
    model.eval()
    test_probs = []
    test_targets = []
    test_loss = 0.0
    n_test = 0
    
    for batch in test_loader:
        logits = model(batch.x)
        loss = criterion(logits, batch.y)
        test_loss += float(loss.numpy().flatten()[0])
        
        prob = 1.0 / (1.0 + np.exp(-logits.numpy()))
        test_probs.append(prob.flatten())
        test_targets.append(batch.y.numpy().flatten())
        n_test += 1
    
    test_probs = np.concatenate(test_probs)
    test_targets = np.concatenate(test_targets)
    test_metrics = compute_metrics(test_targets, test_probs)
    
    print(f"  Test Loss:      {test_loss/max(n_test,1):.4f}")
    print(f"  Accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"  Precision:      {test_metrics['precision']:.4f}")
    print(f"  Recall:         {test_metrics['recall']:.4f}")
    print(f"  F1 Score:       {test_metrics['f1']:.4f}")
    print(f"  AUC (ROC):      {test_metrics['auc']:.4f}")
    print(f"  Confusion:      TP={test_metrics['tp']}, TN={test_metrics['tn']}, "
          f"FP={test_metrics['fp']}, FN={test_metrics['fn']}")
    
    # ── Sample predictions ──
    print("\n" + "-" * 60)
    print("Sample predictions (first 10 test samples):")
    print("-" * 60)
    for i in range(min(10, len(test_probs))):
        pred_label = 1 if test_probs[i] >= 0.5 else 0
        correct = "✓" if pred_label == test_targets[i] else "✗"
        print(f"  [{correct}] prob={test_probs[i]:.4f} → pred={pred_label}, actual={int(test_targets[i])}")
    
    print("\n" + "=" * 60)
    print(" Pipeline complete! torch_lite works end-to-end ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
