import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def eval_epoch(self, dataloader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                preds = torch.sigmoid(outputs) if outputs.dim() == 1 else outputs
                all_preds.append(preds.detach().cpu())
                all_labels.append(y.detach().cpu())
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        return self.evaluate_metrics(all_preds, all_labels)

    def evaluate_metrics(self, preds, labels):
        pred_labels = (preds > 0.5).astype(int)
        acc = accuracy_score(labels, pred_labels)
        f1 = f1_score(labels, pred_labels)
        auc = roc_auc_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'auc': auc}
