from __future__ import annotations

from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from deslib.des.des_clustering import DESClustering
from deslib.des.des_knn import DESKNN
from deslib.des.des_mi import DESMI
from deslib.des.des_p import DESP
from deslib.des.knop import KNOP
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.meta_des import METADES
from deslib.des.probabilistic.deskl import DESKL
from deslib.des.probabilistic.rrc import RRC

from .methods import DESFH, DES_AS, GNNDES, IncA_DES, imDEF


def train_base_pool(templates, X_train, y_train):
    pool = []
    for _, estimator in templates.items():
        model = clone(estimator)
        model.fit(X_train, y_train)
        pool.append(model)
    return pool


def build_static_models(templates):
    estimators_hard = [(name, clone(clf)) for name, clf in templates.items()]
    estimators_soft = [(name, clone(clf)) for name, clf in templates.items()]
    estimators_stack = [(name, clone(clf)) for name, clf in templates.items()]
    return {
        "Voting_Hard": VotingClassifier(estimators=estimators_hard, voting="hard"),
        "Voting_Soft": VotingClassifier(estimators=estimators_soft, voting="soft"),
        "Stacking": StackingClassifier(
            estimators=estimators_stack,
            final_estimator=LogisticRegression(max_iter=1000),
            passthrough=False,
        ),
    }


def build_dynamic_models(trained_pool, k: int = 7, seed: int = 42):
    """Instantiate DES methods using the shared pre-trained classifier pool."""
    return {
        "KNOP": KNOP(trained_pool, k=k),
        "DESClustering": DESClustering(trained_pool),
        "DESKNN": DESKNN(trained_pool, k=k),
        "KNORAE": KNORAE(trained_pool, k=k),
        "KNORAU": KNORAU(trained_pool, k=k),
        "FIRE-DES": KNORAU(trained_pool, k=k, DFP=True),
        "DESMI": DESMI(trained_pool, k=k),
        "RRC": RRC(trained_pool, k=k),
        "DESKL": DESKL(trained_pool, k=k),
        "DESP": DESP(trained_pool, k=k),
        "METADES": METADES(trained_pool, k=k),
        "GNNDES": GNNDES(trained_pool, k=k, seed=seed),
        "DESFH": DESFH(trained_pool, k=k),
        "imDEF": imDEF(trained_pool),
        "DES_AS": DES_AS(trained_pool, k=k, random_state=seed),
        # Source-faithful constructor. The uploaded implementation is stream-oriented
        # and does not expose sklearn-style fit(); runner.py handles partial_fit.
        "IncA_DES": IncA_DES(k=k),
    }
