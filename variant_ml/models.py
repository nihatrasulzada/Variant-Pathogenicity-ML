"""
ADDIM 3: Modelləşdirmə və təlim.

  - 80/20 stratified bölgü
  - Random Forest və XGBoost — hər biri Pipeline(imputer -> classifier) kimi
  - RandomizedSearchCV + StratifiedKFold ilə hiperparametr tənzimləməsi
    (tənzimləmə YALNIZ təlim datasında aparılır; test datası son qiymətləndirmə
     üçün toxunulmaz saxlanılır)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = config.TEST_SIZE,
               random_state: int = config.RANDOM_STATE):
    """Sinif nisbətini qoruyan 80/20 bölgü."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    print(f"[split] Təlim: {len(X_train):,} | Sınaq: {len(X_test):,} | "
          f"Pathogenic nisbəti (təlim/sınaq): {y_train.mean():.3f} / {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def _imputer() -> SimpleImputer:
    # median: skorların paylanması əyridir (SIFT/PolyPhen 0-1 sərhədlərində toplanır)
    # add_indicator: "skor yoxdur" faktının özü də informativ ola bilər
    # keep_empty_features: real ClinVar-da skor sütunu tam boş olsa belə sıra pozulmasın
    return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)


def build_random_forest() -> tuple[Pipeline, dict]:
    pipe = Pipeline([
        ("imputer", _imputer()),
        ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                       random_state=config.RANDOM_STATE)),
    ])
    params = {
        "clf__n_estimators": randint(200, 600),
        "clf__max_depth": [None, 8, 12, 16, 24],
        "clf__min_samples_split": randint(2, 12),
        "clf__min_samples_leaf": randint(1, 6),
        "clf__max_features": ["sqrt", "log2", 0.5],
    }
    return pipe, params


def build_xgboost(y_train: pd.Series) -> tuple[Pipeline, dict]:
    # Sinif disbalansı üçün çəki: (#benign / #pathogenic)
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    pipe = Pipeline([
        ("imputer", _imputer()),
        ("clf", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              tree_method="hist", scale_pos_weight=pos_weight,
                              n_jobs=-1, random_state=config.RANDOM_STATE)),
    ])
    params = {
        "clf__n_estimators": randint(200, 700),
        "clf__max_depth": randint(3, 9),
        "clf__learning_rate": uniform(0.02, 0.18),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__min_child_weight": randint(1, 8),
        "clf__gamma": uniform(0, 2),
        "clf__reg_lambda": uniform(0.5, 4.5),
    }
    return pipe, params


def tune_and_train(name: str, pipe: Pipeline, params: dict,
                   X_train: pd.DataFrame, y_train: pd.Series,
                   n_iter: int = 20, cv_folds: int = 5) -> Pipeline:
    """RandomizedSearchCV ilə tənzimləyir, ən yaxşı modeli bütün təlim datasında yenidən öyrədir."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    search = RandomizedSearchCV(pipe, params, n_iter=n_iter, scoring="roc_auc", cv=cv,
                                n_jobs=-1, random_state=config.RANDOM_STATE,
                                refit=True, verbose=0)
    search.fit(X_train, y_train)
    best = {k.replace("clf__", ""): (round(float(v), 4) if isinstance(v, float) else v)
            for k, v in search.best_params_.items()}
    print(f"[train] {name}: ən yaxşı CV ROC-AUC = {search.best_score_:.4f}")
    print(f"[train] {name}: parametrlər = {best}")
    return search.best_estimator_


def train_all(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 20) -> dict:
    rf_pipe, rf_params = build_random_forest()
    xgb_pipe, xgb_params = build_xgboost(y_train)
    return {
        "Random Forest": tune_and_train("Random Forest", rf_pipe, rf_params,
                                        X_train, y_train, n_iter),
        "XGBoost": tune_and_train("XGBoost", xgb_pipe, xgb_params,
                                  X_train, y_train, n_iter),
    }


def get_feature_names(model: Pipeline, input_columns: list[str]) -> np.ndarray:
    """İmputer-in əlavə etdiyi 'missingindicator_*' sütunları ilə birlikdə adlar."""
    return model.named_steps["imputer"].get_feature_names_out(input_columns)
