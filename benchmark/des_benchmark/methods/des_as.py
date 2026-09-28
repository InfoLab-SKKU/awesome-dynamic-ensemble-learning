# coding=utf-8
import numpy as np
from sklearn.neighbors import NearestNeighbors
from itertools import permutations

class DES_AS:
    """
    Dynamic Ensemble Selection based on Algorithmic Shapley (DES-AS)
    """

    def __init__(self, pool_classifiers, k=7, random_state=None):
        """
        Parameters
        ----------
        pool_classifiers : list
            List of base classifiers.
        k : int
            Number of neighbors for the Region of Competence (RoC).
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.pool_classifiers = pool_classifiers
        self.k = k
        self.random_state = random_state
        self.DSEL_data_ = None
        self.DSEL_labels_ = None
        self.n_classifiers_ = len(pool_classifiers)
        self.classes_ = None

    def fit(self, X_dsel, y_dsel):
        """
        Store the DSEL dataset for RoC computation.
        """
        self.DSEL_data_ = X_dsel
        self.DSEL_labels_ = y_dsel
        self.classes_ = np.unique(y_dsel)
        return self

    def _compute_roc(self, x):
        """
        Compute the Region of Competence (RoC) for a single sample x.
        """
        knn = NearestNeighbors(n_neighbors=self.k)
        knn.fit(self.DSEL_data_)
        distances, indices = knn.kneighbors(x.reshape(1, -1))
        return indices[0]

    def _compute_shapley(self, x, roc_indices):
        """
        Compute Algorithmic Shapley values for all classifiers for a single sample.
        """
        n_clf = self.n_classifiers_
        shapley_values = np.zeros(n_clf)

        # Collect predictions on RoC
        roc_X = self.DSEL_data_[roc_indices]
        roc_y = self.DSEL_labels_[roc_indices]

        # Evaluate all permutations (for small n_clf)
        for pi in permutations(range(n_clf)):
            current_set = []
            prev_score = 0
            for idx in pi:
                current_set.append(idx)
                # Weighted vote of current set
                pred_matrix = np.array([self.pool_classifiers[c].predict(roc_X) for c in current_set]).T
                # Majority vote
                maj_vote = np.array([np.bincount(row, minlength=len(self.classes_)).argmax() for row in pred_matrix])
                score = np.mean(maj_vote == roc_y)
                # Marginal contribution
                marginal = score - prev_score
                if idx in current_set:
                    shapley_values[idx] += marginal
                prev_score = score

        # Average over permutations
        shapley_values /= np.math.factorial(n_clf)
        return shapley_values

    def predict(self, X):
        """
        Predict labels for X.
        """
        proba = self.predict_proba(X)
        y_pred = self.classes_[np.argmax(proba, axis=1)]
        return y_pred

    def predict_proba(self, X):
        """
        Predict probabilities for X.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        P = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            roc_indices = self._compute_roc(x)
            shapley = self._compute_shapley(x, roc_indices)
            # Select only positive Shapley values
            positive_idx = np.where(shapley > 0)[0]
            if len(positive_idx) == 0:
                # fallback: equal weights
                positive_idx = np.arange(self.n_classifiers_)
                weights = np.ones(self.n_classifiers_) / self.n_classifiers_
            else:
                weights = shapley[positive_idx] / np.sum(shapley[positive_idx])

            # Weighted voting
            weighted_votes = np.zeros((n_classes,))
            for w, clf_idx in zip(weights, positive_idx):
                pred = self.pool_classifiers[clf_idx].predict(x.reshape(1, -1))[0]
                weighted_votes[np.where(self.classes_ == pred)[0][0]] += w
            P[i] = weighted_votes

        # Normalize
        P /= P.sum(axis=1, keepdims=True)
        return P
