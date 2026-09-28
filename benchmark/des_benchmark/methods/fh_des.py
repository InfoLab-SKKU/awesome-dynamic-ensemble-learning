# des_fh_predict.py
# coding=utf-8
# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
# FH-DES with predict and predict_proba support

import numpy as np
from deslib.des.base import BaseDES
from typing import Optional, Sequence

class Hyperbox:
    def __init__(self, v, w, classifier, theta=0.05):
        self.Min = np.minimum(v, w)
        self.Max = np.maximum(v, w)
        self.clsr = classifier
        self.theta = theta
        self.Center = (self.Min + self.Max) / 2

    def is_expandable(self, x):
        return np.all((np.abs(x - self.Center)) <= self.theta)

    def expand(self, x):
        self.Min = np.minimum(self.Min, x)
        self.Max = np.maximum(self.Max, x)
        self.Center = (self.Min + self.Max) / 2
        

class DESFH(BaseDES):
    """
    Fuzzy Hyperbox Dynamic Ensemble Selection (DES-FH)
    with predict and predict_proba methods.
    """

    def __init__(
        self,
        pool_classifiers: Optional[Sequence] = None,
        k: int = 7,
        DFP: bool = False,
        with_IH: bool = False,
        safe_k: Optional[int] = None,
        IH_rate: float = 0.30,
        random_state: Optional[int] = None,
        knn_classifier: str = "knn",
        DSEL_perc: float = 0.5,
        HyperBoxes: Optional[Sequence[Hyperbox]] = None,
        theta: float = 0.05,
        mu: float = 0.991,
        mis_sample_based: bool = True,
    ):
        self.theta = theta
        self.mu = mu
        self.mis_sample_based = mis_sample_based
        self.HBoxes = list(HyperBoxes) if HyperBoxes is not None else []
        self.NO_hypeboxes = len(self.HBoxes)

        super(DESFH, self).__init__(
            pool_classifiers=pool_classifiers,
            with_IH=with_IH,
            safe_k=safe_k,
            IH_rate=IH_rate,
            mode="hybrid",
            random_state=random_state,
            DSEL_perc=DSEL_perc,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        super(DESFH, self).fit(X, y)

        if not (0 < self.mu <= 1):
            raise ValueError("The value of mu must be in (0,1].")
        if not (0 < self.theta <= 1):
            raise ValueError("The value of theta must be in (0,1].")

        if self.mis_sample_based:
            for classifier_index in range(self.n_classifiers_):
                MissSet_indexes = ~self.DSEL_processed_[:, classifier_index]
                self.setup_hyperboxs(MissSet_indexes, classifier_index)
        else:
            for classifier_index in range(self.n_classifiers_):
                WellSet_indexes = self.DSEL_processed_[:, classifier_index]
                self.setup_hyperboxs(WellSet_indexes, classifier_index)

        self.NO_hypeboxes = len(self.HBoxes)
        return self

    def estimate_competence(self, query: np.ndarray, neighbors=None, distances=None, predictions=None) -> np.ndarray:
        q = np.asarray(query)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        n_queries, n_features = q.shape

        if self.NO_hypeboxes == 0:
            if self.mis_sample_based:
                return np.ones((n_queries, self.n_classifiers_))
            else:
                return np.zeros((n_queries, self.n_classifiers_))

        nb = len(self.HBoxes)
        boxes_classifier = np.zeros((nb,), dtype=int)
        boxes_W = np.zeros((nb, n_features), dtype=float)
        boxes_V = np.zeros((nb, n_features), dtype=float)
        boxes_center = np.zeros((nb, n_features), dtype=float)

        for i, box in enumerate(self.HBoxes):
            boxes_classifier[i] = int(box.clsr)
            boxes_W[i, :] = box.Max
            boxes_V[i, :] = box.Min
            boxes_center[i, :] = (box.Max + box.Min) / 2.0

        if self.mis_sample_based:
            competences_ = np.ones((n_queries, self.n_classifiers_), dtype=float)
        else:
            competences_ = np.zeros((n_queries, self.n_classifiers_), dtype=float)

        boxes_W_b = boxes_W.reshape(nb, 1, n_features)
        boxes_V_b = boxes_V.reshape(nb, 1, n_features)
        boxes_center_b = boxes_center.reshape(nb, 1, n_features)
        Xq = q.reshape(1, n_queries, n_features)

        halfsize = (boxes_W_b - boxes_V_b) / 2.0
        d = np.abs(boxes_center_b - Xq) - halfsize
        d[d < 0] = 0.0
        dd = np.linalg.norm(d, axis=2)
        dd = dd / np.sqrt(n_features)
        m = np.clip(1.0 - dd, 0.0, 1.0)
        m = np.power(m, 4.0)

        classifiers_unique, indices, counts = np.unique(boxes_classifier, return_index=True, return_counts=True)
        for idx_cls_pos, clsr in enumerate(classifiers_unique):
            start = indices[idx_cls_pos]
            cnt = counts[idx_cls_pos]
            c_range = range(start, start + cnt)
            cmat = m[list(c_range), :]

            if cnt > 1:
                bb_indexes = np.argsort(-cmat, axis=0)
                b1 = bb_indexes[0, :]
                b2 = bb_indexes[1, :]
                for qi in range(n_queries):
                    val_b1 = cmat[b1[qi], qi]
                    val_b2 = cmat[b2[qi], qi]
                    competences_[qi, int(clsr)] = 0.7 * val_b1 + 0.3 * val_b2
            else:
                for qi in range(n_queries):
                    competences_[qi, int(clsr)] = cmat[0, qi]

        return competences_

    def setup_hyperboxs(self, samples_ind, classifier: int):
        if samples_ind is None:
            return
        samples_ind = np.asarray(samples_ind)
        if samples_ind.size == 0:
            return

        if samples_ind.dtype == bool:
            if not np.any(samples_ind):
                return
            selected_samples = self.DSEL_data_[samples_ind, :]
        else:
            if len(samples_ind) == 0:
                return
            selected_samples = self.DSEL_data_[samples_ind, :]

        boxes = []
        for X in selected_samples:
            if len(boxes) == 0:
                b = Hyperbox(v=X.copy(), w=X.copy(), classifier=classifier, theta=self.theta)
                boxes.append(b)
                continue

            IsInBox = False
            for box in boxes:
                if np.all(box.Min < X) and np.all(box.Max > X):
                    IsInBox = True
                    break
            if IsInBox:
                continue

            nDist = np.inf
            nearest_box = None
            for box in boxes:
                dist = np.linalg.norm(X - box.Center)
                if dist < nDist:
                    nearest_box = box
                    nDist = dist

            if nearest_box is not None and nearest_box.is_expandable(X):
                nearest_box.expand(X)
                continue

            b = Hyperbox(v=X.copy(), w=X.copy(), classifier=classifier, theta=self.theta)
            boxes.append(b)

        if len(boxes) > 0:
            self.HBoxes.extend(boxes)
            self.NO_hypeboxes = len(self.HBoxes)

    def select(self, competences: np.ndarray) -> np.ndarray:
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)
        max_value = np.max(competences, axis=1)
        selected_classifiers = competences >= (self.mu * max_value.reshape(-1, 1))
        return selected_classifiers

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X using competence-weighted votes.
        """
        competences = self.estimate_competence(X)
        selected = self.select(competences)
        n_samples = X.shape[0] if X.ndim > 1 else 1
        n_classes = self.n_classes_
        proba = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for clf_idx, clf in enumerate(self.pool_classifiers_):
                if selected[i, clf_idx]:
                    proba[i] += competences[i, clf_idx] * clf.predict_proba(X[i].reshape(1, -1))[0]
            # normalize
            if np.sum(proba[i]) > 0:
                proba[i] /= np.sum(proba[i])
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for X using FH-DES.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
