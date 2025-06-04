from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

def evaluate_binary(preds, labels, threshold=0.5):
    pred_labels = (preds > threshold).astype(int)
    acc = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    try:
        auc = roc_auc_score(labels, preds)
    except:
        auc = float('nan')
    return {'accuracy': acc, 'f1': f1, 'auc': auc}

def evaluate_multiclass(preds, labels):
    pred_labels = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels, average='weighted')
    return {'accuracy': acc, 'f1': f1}
