# gnn_des.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from typing import List
import math

# ----------------------
# Helper: classifier -> probabilities
# ----------------------
def clf_predict_proba(clf, X):
    """
    Return probability matrix (n_samples, n_classes).
    If classifier has predict_proba, use it.
    Otherwise try decision_function and convert to probabilities via softmax.
    """
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        # ensure shape (n_samples, n_classes)
        return proba
    elif hasattr(clf, "decision_function"):
        df = clf.decision_function(X)
        # decision_function may return (n_samples,) for binary, or (n_samples, n_classes)
        if df.ndim == 1:
            # binary -> convert to 2-class logit form
            df = np.vstack([-df, df]).T
        # apply softmax numerically stable
        e = np.exp(df - np.max(df, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    else:
        # fallback: use predict -> one-hot
        preds = clf.predict(X)
        classes = np.unique(preds)
        # build mapping from clf.classes_ if exists
        if hasattr(clf, "classes_"):
            all_classes = clf.classes_
        else:
            all_classes = np.unique(preds)
        n_classes = len(all_classes)
        proba = np.zeros((len(preds), n_classes))
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        for i, p in enumerate(preds):
            proba[i, class_to_idx[p]] = 1.0
        return proba

# ----------------------
# Tiny GCN module (two-layer)
# ----------------------
class TinyGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=1, dropout=0.2):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A_norm):
        """
        X: (num_nodes, in_dim)
        A_norm: (num_nodes, num_nodes) normalized adjacency (torch tensor)
        returns: (num_nodes, out_dim)
        """
        # message: A_norm @ X
        h = torch.matmul(A_norm, X)         # (N, in_dim)
        h = self.lin1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = torch.matmul(A_norm, h)
        h = self.lin2(h)
        return h  # raw scores (N, out_dim)

# ----------------------
# GNNDES class
# ----------------------
class GNNDES:
    """
    Fast GNN-based Dynamic Ensemble Selection.
    No expensive training loop.
    Graph acts as structural smoother over classifier competences.
    """

    def __init__(
        self,
        pool_classifiers,
        k=7,
        hidden_dim=32,
        graph_threshold=0.0,
        device=None,
        seed=42,
    ):
        self.pool = list(pool_classifiers)
        self.n_classifiers = len(self.pool)
        self.k = k
        self.graph_threshold = graph_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.node_feat_dim = 4
        self.gnn = TinyGCN(self.node_feat_dim, hidden_dim=hidden_dim, out_dim=1).to(self.device)

        self.nn_model = None
        self.A_norm = None
        self.global_acc = None
        self.classifier_probas_dsel = None
        self.n_classes = None

    def fit(self, X_dsel, y_dsel):

        self.X_dsel = np.asarray(X_dsel)
        self.y_dsel = np.asarray(y_dsel)
        n_samples = len(self.X_dsel)

        # Precompute classifier probabilities on DSEL
        probs = []
        for clf in self.pool:
            probs.append(clf_predict_proba(clf, self.X_dsel))
        self.classifier_probas_dsel = np.array(probs)
        self.n_classes = self.classifier_probas_dsel.shape[2]

        # Global accuracy
        preds = np.argmax(self.classifier_probas_dsel, axis=2)
        self.global_acc = np.mean(preds == self.y_dsel, axis=1)

        # Build adjacency (correlation-based)
        conf = np.max(self.classifier_probas_dsel, axis=2)
        corr = np.corrcoef(conf)
        corr = np.nan_to_num(corr)
        sim = np.abs(corr)

        if self.graph_threshold > 0:
            sim[sim < self.graph_threshold] = 0.0

        np.fill_diagonal(sim, 1.0)

        # Normalize adjacency
        D = np.sum(sim, axis=1)
        D_inv_sqrt = np.power(D + 1e-12, -0.5)
        D_mat = np.diag(D_inv_sqrt)
        A_norm = D_mat @ sim @ D_mat
        self.A_norm = torch.tensor(A_norm, dtype=torch.float32, device=self.device)

        # kNN model
        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.k, n_samples)
        )
        self.nn_model.fit(self.X_dsel)

        self.gnn.eval()  # no training
        return self

    def _compute_node_features(self, x_query):

        # RoC neighbors
        _, idx = self.nn_model.kneighbors(x_query.reshape(1, -1))
        idx = idx[0]

        # Local accuracy (vectorized)
        local_preds = np.argmax(self.classifier_probas_dsel[:, idx, :], axis=2)
        local_acc = np.mean(local_preds == self.y_dsel[idx], axis=1)

        # Query probabilities (vectorized)
        probs = np.array([
            clf_predict_proba(clf, x_query.reshape(1, -1))[0]
            for clf in self.pool
        ])

        max_prob = np.max(probs, axis=1)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

        feats = np.vstack([
            self.global_acc,
            local_acc,
            max_prob,
            entropy
        ]).T

        feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-12)
        feats = np.nan_to_num(feats)

        return feats, probs

    def predict_proba(self, X):

        X = np.asarray(X)
        results = []

        with torch.no_grad():
            for x in X:
                node_feats, probs_clfs = self._compute_node_features(x)

                X_nodes = torch.tensor(
                    node_feats,
                    dtype=torch.float32,
                    device=self.device
                )

                scores = self.gnn(X_nodes, self.A_norm).squeeze(1)
                weights = torch.softmax(scores, dim=0).cpu().numpy()

                combined = np.sum(
                    probs_clfs * weights.reshape(-1, 1),
                    axis=0
                )

                results.append(combined)

        return np.vstack(results)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    @property
    def classes_(self):
        if hasattr(self.pool[0], "classes_"):
            return self.pool[0].classes_
        return np.arange(self.n_classes)
