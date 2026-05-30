import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class imDEF(BaseEstimator, ClassifierMixin):
    """
    Multi-class imDEF for PRETRAINED classifiers.

    Parameters
    ----------
    pool_classifiers : list
        List of pretrained sklearn classifiers.

    b : int
        Number of hardness bins.

    beta : int
        Number of self-paced iterations.
    """

    def __init__(
        self,
        pool_classifiers,
        b=20,
        beta=10
    ):

        self.pool_classifiers = pool_classifiers
        self.b = b
        self.beta = beta

        self.classes_ = None
        self.weights_ = None
        self.label_encoder_ = LabelEncoder()

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        n_samples = len(X)
        n_clfs = len(self.pool_classifiers)
        n_classes = len(self.classes_)

        # ==========================================================
        # GET PREDICTIONS FROM ALL CLASSIFIERS
        # ==========================================================

        all_proba = np.zeros((n_clfs, n_samples, n_classes))

        for i, clf in enumerate(self.pool_classifiers):

            proba = clf.predict_proba(X)

            # Handle classifiers with missing classes
            aligned = np.zeros((n_samples, n_classes))

            for j, cls in enumerate(clf.classes_):

                global_idx = np.where(self.classes_ == cls)[0][0]
                aligned[:, global_idx] = proba[:, j]

            all_proba[i] = aligned

        # ==========================================================
        # COMPUTE CLASSIFIER ERRORS
        # ==========================================================

        errors = np.zeros((n_clfs, n_samples))

        for i in range(n_clfs):

            preds = np.argmax(all_proba[i], axis=1)

            errors[i] = (preds != y_encoded).astype(float)

        # ==========================================================
        # INITIAL WEIGHTS
        # ==========================================================

        weights = np.ones(n_clfs) / n_clfs

        # ==========================================================
        # SELF-PACED WEIGHTING
        # ==========================================================

        for t in range(1, self.beta + 1):

            weighted_error = np.average(errors, axis=1)

            threshold_bins = np.linspace(
                weighted_error.min(),
                weighted_error.max() + 1e-8,
                self.b + 1
            )

            bin_indices = [
                np.where(
                    (weighted_error >= threshold_bins[i]) &
                    (weighted_error < threshold_bins[i + 1])
                )[0]
                for i in range(self.b)
            ]

            avg_error = np.array([
                weighted_error[idx].mean()
                if len(idx) > 0 else 0
                for idx in bin_indices
            ])

            sp_f = np.tan((t) * np.pi / (2 * self.beta + 1))

            bin_weights = 1 / (sp_f + avg_error + 1e-8)

            if np.sum(bin_weights) > 0:
                bin_weights /= np.sum(bin_weights)

            new_weights = np.zeros(n_clfs)

            for i, idx in enumerate(bin_indices):

                if len(idx) == 0:
                    continue

                new_weights[idx] = bin_weights[i]

            if np.sum(new_weights) > 0:
                new_weights /= np.sum(new_weights)

            weights = new_weights

        self.weights_ = weights

        return self

    def predict_proba(self, X):

        X = np.array(X)

        n_samples = len(X)
        n_classes = len(self.classes_)

        final_proba = np.zeros((n_samples, n_classes))

        # ==========================================================
        # WEIGHTED ENSEMBLE
        # ==========================================================

        for weight, clf in zip(self.weights_, self.pool_classifiers):

            proba = clf.predict_proba(X)

            aligned = np.zeros((n_samples, n_classes))

            for j, cls in enumerate(clf.classes_):

                global_idx = np.where(self.classes_ == cls)[0][0]
                aligned[:, global_idx] = proba[:, j]

            final_proba += weight * aligned

        # Normalize
        sums = final_proba.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1

        final_proba /= sums

        return final_proba

    def predict(self, X):

        probs = self.predict_proba(X)

        preds = np.argmax(probs, axis=1)

        return self.label_encoder_.inverse_transform(preds)