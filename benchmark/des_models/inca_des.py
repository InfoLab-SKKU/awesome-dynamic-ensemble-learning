# coding=utf-8
import numpy as np
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from collections import deque

def canberra_distance(a, b):
    """Compute Canberra distance between two arrays"""
    return np.sum(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12))

class OnlineKDTree:
    """
    Simple online K-d tree using dynamic arrays and cKDTree rebuilds.
    Supports insertion/removal for streaming data.
    """
    def __init__(self, data=None):
        self.data = np.array(data) if data is not None else np.empty((0,0))
        self.tree = None
        if len(self.data) > 0:
            self.rebuild_tree()
    
    def rebuild_tree(self):
        if len(self.data) > 0:
            self.tree = cKDTree(self.data)
        else:
            self.tree = None

    def insert(self, x):
        self.data = np.vstack([self.data, x])
        self.rebuild_tree()

    def remove_index(self, idx):
        self.data = np.delete(self.data, idx, axis=0)
        self.rebuild_tree()

    def query(self, x, k=1):
        if self.tree is None:
            return [], []
        dists, idxs = self.tree.query(x, k=min(k, len(self.data)))
        return idxs, dists


class IncA_DES:
    """
    Incremental Adaptive Dynamic Ensemble Selection (IncA-DES)
    """
    def __init__(self, base_classifier=DecisionTreeClassifier, k=7, W=200, F=50, mu=0.8):
        """
        Parameters
        ----------
        base_classifier : sklearn estimator class
            Classifier type for ensemble members.
        k : int
            Number of neighbors for RoC.
        W : int
            Max ensemble window size (DSEW).
        F : int
            Max instances per classifier before adding a new one.
        mu : float
            Competence threshold multiplier.
        """
        self.base_classifier = base_classifier
        self.k = k
        self.W = W
        self.F = F
        self.mu = mu

        self.ensemble = []            # list of classifiers
        self.classifier_instance_count = []  # track instances per classifier
        self.DSEW = deque(maxlen=W)   # ensemble window (instances)
        self.DSEW_X = []               # features of DSEW
        self.DSEW_y = []               # labels of DSEW
        self.kd_tree = None
        self.classes_ = None

    def _add_instance_to_window(self, x, y):
        """Add a new instance to DSEW and update KD-tree."""
        if len(self.DSEW_X) >= self.W:
            # Remove oldest
            self.DSEW_X.pop(0)
            self.DSEW_y.pop(0)
        self.DSEW.append((x, y))
        self.DSEW_X.append(x)
        self.DSEW_y.append(y)
        self.kd_tree = OnlineKDTree(np.array(self.DSEW_X))

    def _select_classifier(self, x):
        """Compute competence and select classifiers for x"""
        selected = []
        weights = []

        if not self.ensemble:
            return selected, weights

        # Compute competence per classifier
        X_query = np.array(x).reshape(1, -1)
        roc_idx, _ = self.kd_tree.query(X_query, k=min(self.k, len(self.DSEW_X)))
        roc_X = np.array([self.DSEW_X[i] for i in roc_idx])
        roc_y = np.array([self.DSEW_y[i] for i in roc_idx])

        competences = []
        for clf in self.ensemble:
            preds = clf.predict(roc_X)
            competence = np.mean(preds == roc_y)
            competences.append(competence)
        competences = np.array(competences)

        # Select classifiers above threshold
        tau = self.mu * np.max(competences)
        selected_idx = np.where(competences >= tau)[0]
        selected = [self.ensemble[i] for i in selected_idx]
        weights = competences[selected_idx] / np.sum(competences[selected_idx]) if len(selected_idx) > 0 else []
        return selected, weights

    def partial_fit(self, X, y):
        """
        Incremental training for each instance
        """
        if self.classes_ is None:
            self.classes_ = np.unique(y)

        for xi, yi in zip(X, y):
            # Add instance to ensemble window
            self._add_instance_to_window(xi, yi)

            # Determine if new classifier needed
            if not self.ensemble or (len(self.classifier_instance_count) > 0 and 
                                     self.classifier_instance_count[-1] >= self.F):
                clf = self.base_classifier()
                self.ensemble.append(clf)
                self.classifier_instance_count.append(0)
            
            # Train latest classifier on new instance
            clf = self.ensemble[-1]
            if not hasattr(clf, "classes_"):
                clf.partial_fit(np.array([xi]), np.array([yi]), classes=self.classes_)
            else:
                clf.partial_fit(np.array([xi]), np.array([yi]))
            self.classifier_instance_count[-1] += 1

    def predict_proba(self, X):
        """
        Predict probabilities using selected classifiers and weighted voting
        """
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        P = np.zeros((n_samples, n_classes))

        for i, xi in enumerate(X):
            selected, weights = self._select_classifier(xi)
            if len(selected) == 0:
                # fallback: uniform vote
                P[i] = np.ones(n_classes) / n_classes
                continue
            vote = np.zeros(n_classes)
            for w, clf in zip(weights, selected):
                pred = clf.predict(np.array([xi]))[0]
                vote[np.where(self.classes_ == pred)[0][0]] += w
            P[i] = vote / np.sum(vote)

        return P

    def predict(self, X):
        P = self.predict_proba(X)
        return self.classes_[np.argmax(P, axis=1)]
