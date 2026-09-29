from __future__ import annotations

from collections import OrderedDict

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

DEFAULT_SEEDS = [0, 42, 123, 2021, 7]
DEFAULT_K = 7

DATASETS = OrderedDict([
    ("Wine Quality Red", "wine_quality_red"),
    ("Wine Quality White", "wine_quality_white"),
    ("Led 24", "led24"),
    ("Led 7", "led7"),
    ("Yeast", "yeast"),
    ("ecoli", "ecoli"),
    ("vehicle", "vehicle"),
    ("Satimage", "satimage"),
    ("adult", "adult"),
    ("glass", "glass2"),
    ("wine_recognition", "wine_recognition"),
    ("collins", "collins"),
    ("soybean", "soybean"),
    ("penguins", "penguins"),
    ("heart_statlog", "saheart"),
    ("ionosphere", "ionosphere"),
    ("sonar", "sonar"),
    ("phoneme", "phoneme"),
    ("texture", "texture"),
    ("car", "car_evaluation"),
    ("breast", "breast_cancer"),
    ("dna", "dna"),
    ("shuttle", "shuttle"),
    ("magic", "magic"),
    ("fars", "fars"),
])

DESLIB_METHODS = [
    "KNOP", "DESClustering", "DESKNN", "KNORAE", "KNORAU", "FIRE-DES",
    "DESMI", "RRC", "DESKL", "DESP", "METADES",
]
CUSTOM_METHODS = ["GNNDES", "DESFH", "imDEF", "DES_AS", "IncA_DES"]
STATIC_METHODS = ["Voting_Hard", "Voting_Soft", "Stacking"]
ALL_METHODS = DESLIB_METHODS + CUSTOM_METHODS + STATIC_METHODS


def make_base_classifiers(profile: str = "current13", random_state: int = 42):
    """Create fresh base-classifier templates.

    profile='current13' reproduces the classifier dictionary in the uploaded notebook.
    profile='full17' adds the four Naive Bayes estimators that were imported in the
    notebook but omitted from that dictionary. Use the profile that matches the
    manuscript/results you intend to reproduce.
    """
    clfs = OrderedDict([
        ("RF", RandomForestClassifier(random_state=random_state)),
        ("GB", GradientBoostingClassifier(random_state=random_state)),
        ("ET", ExtraTreesClassifier(random_state=random_state)),
        # Kept source-faithful to the uploaded notebook: these three labels use
        # HistGradientBoostingClassifier there. Rename/replace only if your actual
        # benchmark used the external XGBoost/CatBoost/LightGBM libraries.
        ("XGB", XGBClassifier(random_state=random_state, verbosity=0)),
        ("CatBoost", CatBoostClassifier(random_state=random_state, verbose=False)),
        ("LGBM", LGBMClassifier(random_state=random_state, verbosity=-1)),
        ("Bagging", BaggingClassifier(random_state=random_state)),
        ("AdaBoost", AdaBoostClassifier(random_state=random_state)),
        ("SVC", SVC(probability=True, random_state=random_state)),
        ("KNN", KNeighborsClassifier()),
        ("LR", LogisticRegression(max_iter=1000, random_state=random_state)),
        ("DT", DecisionTreeClassifier(random_state=random_state)),
        ("MLPClassifier", MLPClassifier(max_iter=500, random_state=random_state)),
    ])
    if profile == "full17":
        clfs.update([
            ("GaussianNB", GaussianNB()),
            ("BernoulliNB", BernoulliNB()),
            ("MultinomialNB", MultinomialNB()),
            ("ComplementNB", ComplementNB()),
        ])
    elif profile != "current13":
        raise ValueError("profile must be 'current13' or 'full17'")
    return clfs
