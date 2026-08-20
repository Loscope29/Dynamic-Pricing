"""Modele de demande : l'objet qui repond a "combien j'en vends a ce prix ?".

Le modele predit `log(quantite)` puis revient a l'echelle des unites. Ce
detour n'est pas anodin : `exp(E[log Q])` n'est PAS `E[Q]` (inegalite de
Jensen), et l'ignorer sous-estime systematiquement la demande d'environ
`exp(sigma^2/2)`. On applique donc la **correction de Duan (smearing)**,
estimee sur les residus d'entrainement -- non parametrique, donc valable meme
si les residus ne sont pas gaussiens.

L'interface publique se resume a `predict_demand(df) -> unites`, parce que
c'est exactement ce dont l'optimiseur a besoin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from optimization_pricing.config import (
    CATEGORICAL_FEATURES,
    CATEGORY_COL,
    FEATURES,
    NUMERIC_FEATURES,
)


@dataclass
class DemandModel:
    """Enveloppe unifiee autour d'un estimateur scikit-learn ou LightGBM."""

    estimator: object
    name: str = "demand_model"
    uses_native_categorical: bool = False
    categories: list[str] = field(default_factory=list)
    smearing: float = 1.0
    metadata: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- #
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Met les colonnes dans l'ordre et le type attendus par l'estimateur.

        L'ordre des colonnes est critique pour LightGBM : les contraintes de
        monotonie sont positionnelles. Une permutation silencieuse
        appliquerait la contrainte "la demande decroit" a la mauvaise
        variable.
        """
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise KeyError(f"Features absentes a la prediction : {missing}")

        X = df[FEATURES].copy()
        if self.uses_native_categorical:
            X[CATEGORY_COL] = pd.Categorical(X[CATEGORY_COL], categories=self.categories)
        return X

    # ---------------------------------------------------------------- #
    def fit(self, df: pd.DataFrame, target: str = "log_quantity") -> DemandModel:
        if self.uses_native_categorical and not self.categories:
            self.categories = sorted(df[CATEGORY_COL].astype(str).unique())
        X, y = self._prepare(df), df[target].to_numpy()
        self.estimator.fit(X, y)

        # Facteur de smearing de Duan : moyenne de exp(residus) sur le train
        residuals = y - self.estimator.predict(X)
        self.smearing = float(np.mean(np.exp(residuals)))
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(self._prepare(df)), dtype=float)

    def predict_demand(self, df: pd.DataFrame) -> np.ndarray:
        """Demande esperee en unites, correction de retransformation incluse."""
        return np.exp(self.predict_log(df)) * self.smearing

    # ---------------------------------------------------------------- #
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: Path) -> DemandModel:
        with open(Path(path), "rb") as f:
            return pickle.load(f)


# --------------------------------------------------------------------------- #
# Fabriques d'estimateurs
# --------------------------------------------------------------------------- #
def build_sklearn_pipeline(estimator):
    """Pipeline sklearn : one-hot sur la categorie, numerique inchange."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("pre", pre), ("model", estimator)])
