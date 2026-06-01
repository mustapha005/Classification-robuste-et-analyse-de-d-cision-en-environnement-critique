"""
Projet IA - Classification robuste et analyse de decision en environnement critique.

Ce script couvre toutes les parties demandees dans l'enonce:
1) EDA + correlation + VIF + desequilibre.
2) Comparaison class_weight vs SMOTE/ADASYN.
3) Regression Logistique Elastic Net.
4) Random Forest + matrice de proximite + outliers de prediction.
5) XGBoost cost-sensitive avec scale_pos_weight et loss customisee.
6) Optimisation Optuna avec graphiques de convergence.
7) Evaluation avancee: F1-macro, AUPRC, MCC, courbe Precision-Recall, cout.
8) Calibration: reliability diagrams + Platt Scaling ou Isotonic Regression.
9) Interpretabilite SHAP.

Execution exemple:
python src/project_ai_final.py --dataset creditcard --data_path data/creditcard.csv --n_trials 30
python src/project_ai_final.py --dataset aps --n_trials 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


@dataclass
class CostConfig:
    """Couts asymetriques pour la decision.

    fp: cout d'un faux positif.
    fn: cout d'un faux negatif.
    Pour APS, l'enonce original UCI donne fp=10 et fn=500.
    Pour fraude bancaire, on choisit par defaut fp=1 et fn=50 car rater une fraude
    est beaucoup plus grave que verifier une transaction normale.
    """

    fp: float
    fn: float


# ---------------------------------------------------------------------------
# 0. Outils generaux
# ---------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def decision_threshold_from_cost(cost: CostConfig) -> float:
    """Seuil theorique si les probabilites sont bien calibrees.

    On predit positif si: P(y=1|x) * cout_FN > P(y=0|x) * cout_FP
    donc seuil = cout_FP / (cout_FP + cout_FN).
    """
    return cost.fp / (cost.fp + cost.fn)


def optimize_threshold_by_mcc(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
    """Cherche le seuil qui maximise MCC sur un jeu de validation."""
    thresholds = np.unique(np.quantile(proba, np.linspace(0.01, 0.99, 99)))
    best_t, best_mcc = 0.5, -1.0
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        if score > best_mcc:
            best_t, best_mcc = float(t), float(score)
    return best_t, best_mcc


def compute_cost(y_true: np.ndarray, y_pred: np.ndarray, cost: CostConfig) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(cost.fp * fp + cost.fn * fn)


def evaluate_binary_model(
    name: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    cost: CostConfig,
    out_dir: Path,
) -> Dict[str, float]:
    """Evaluation sans accuracy: F1-macro, AUPRC, MCC, Brier, LogLoss et cout."""
    proba = np.asarray(proba).reshape(-1)
    proba = np.clip(proba, 1e-8, 1 - 1e-8)
    y_pred = (proba >= threshold).astype(int)

    metrics = {
        "model": name,
        "threshold": float(threshold),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "auprc_average_precision": float(average_precision_score(y_true, proba)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier_loss": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "business_cost": compute_cost(y_true, y_pred, cost),
    }

    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)
    print("Threshold:", threshold)
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix [tn fp; fn tp]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(metrics)

    precision, recall, _ = precision_recall_curve(y_true, proba)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"pr_curve_{safe_name(name)}.png", dpi=180)
    plt.close()

    return metrics


def safe_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


# ---------------------------------------------------------------------------
# 1. Chargement des bases recommandees
# ---------------------------------------------------------------------------


def load_creditcard(data_path: str | Path) -> Tuple[pd.DataFrame, pd.Series, CostConfig]:
    """Charge Kaggle Credit Card Fraud Detection.

    Fichier attendu: creditcard.csv avec colonnes Time, V1...V28, Amount, Class.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable: {data_path}. Telechargez-le avec:\n"
            "kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip"
        )

    df = pd.read_csv(data_path)
    expected = {"Time", "Amount", "Class"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour Credit Card: {missing}")

    # Feature engineering adapte au dataset fraude:
    # Les variables V1...V28 sont deja issues d'une PCA. On enrichit Time et Amount.
    df = df.copy()
    df["Hour"] = ((df["Time"] / 3600.0) % 24).astype(float)
    df["LogAmount"] = np.log1p(df["Amount"].astype(float))
    df["Amount_to_median"] = df["Amount"] / (df["Amount"].median() + 1e-9)

    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])

    # Couts pedagogiques pour fraude: rater une fraude coute beaucoup plus cher.
    cost = CostConfig(fp=1.0, fn=50.0)
    return X, y, cost


def load_aps() -> Tuple[pd.DataFrame, pd.Series, CostConfig]:
    """Charge APS Failure at Scania Trucks depuis UCI.

    Classes: neg -> 0, pos -> 1.
    Couts UCI: FP = 10, FN = 500.
    """
    from ucimlrepo import fetch_ucirepo

    aps = fetch_ucirepo(id=421)
    X = aps.data.features.copy()
    y_raw = aps.data.targets.iloc[:, 0].copy()
    y = y_raw.map({"neg": 0, "pos": 1}).astype(int)

    # Les valeurs manquantes peuvent arriver sous forme 'na'.
    X = X.replace("na", np.nan)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    cost = CostConfig(fp=10.0, fn=500.0)
    return X, y, cost


# ---------------------------------------------------------------------------
# 2. EDA, correlation, VIF, desequilibre
# ---------------------------------------------------------------------------


def run_eda(X: pd.DataFrame, y: pd.Series, out_dir: Path, vif_max_features: int = 40) -> None:
    """Analyse exploratoire: distribution, correlation, VIF."""
    summary = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "positive_rate": float(y.mean()),
        "missing_rate_mean": float(X.isna().mean().mean()),
        "missing_rate_max": float(X.isna().mean().max()),
    }
    save_json(summary, out_dir / "eda_summary.json")
    print("\nEDA summary:")
    print(json.dumps(summary, indent=2))

    # Distribution des classes.
    counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Classe 0", "Classe 1"], [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Distribution des classes")
    plt.ylabel("Nombre d'observations")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=180)
    plt.close()

    # Matrice de correlation: on limite aux variables les plus informatives pour lisibilite.
    corr_target = X.copy()
    corr_target["target"] = y.values
    corr_with_target = corr_target.corr(numeric_only=True)["target"].abs().sort_values(ascending=False)
    top_cols = [c for c in corr_with_target.index if c != "target"][:25]
    corr = X[top_cols].corr(numeric_only=True)

    plt.figure(figsize=(11, 9))
    im = plt.imshow(corr, interpolation="nearest", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(top_cols)), top_cols, rotation=90, fontsize=7)
    plt.yticks(range(len(top_cols)), top_cols, fontsize=7)
    plt.title("Matrice de correlation - top variables liees a la cible")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_matrix.png", dpi=180)
    plt.close()

    # VIF: tres couteux si trop de colonnes. On garde les variables les plus correlees a la cible.
    vif_cols = top_cols[:vif_max_features]
    X_vif = X[vif_cols].replace([np.inf, -np.inf], np.nan)
    X_vif = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X_vif),
        columns=vif_cols,
    )
    X_vif = X_vif.loc[:, X_vif.std(axis=0) > 0]

    vif_rows = []
    for i, col in enumerate(X_vif.columns):
        try:
            vif_value = variance_inflation_factor(X_vif.values, i)
        except Exception:
            vif_value = np.nan
        vif_rows.append({"feature": col, "vif": float(vif_value) if np.isfinite(vif_value) else np.nan})
    vif_df = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)
    vif_df.to_csv(out_dir / "vif.csv", index=False)


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------


def build_linear_preprocessor(k_best: int = 25) -> Pipeline:
    """Preprocessing pour modele lineaire.

    - Imputation mediane.
    - Standardisation necessaire pour Elastic Net.
    - Selection statistique par information mutuelle.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=mutual_info_classif, k=k_best)),
        ]
    )


def build_tree_preprocessor() -> Pipeline:
    """Preprocessing pour arbres: imputation mediane, pas besoin de scaling."""
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])


# ---------------------------------------------------------------------------
# 4. Traitement du desequilibre: algorithmique vs donnees
# ---------------------------------------------------------------------------


def train_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cost: CostConfig,
    out_dir: Path,
) -> Tuple[Dict[str, Pipeline], List[Dict[str, float]]]:
    """Compare class_weight, SMOTE et ADASYN sur une Regression Logistique Elastic Net."""
    k_best = min(25, X_train.shape[1])
    base_lr = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=0.1,
        max_iter=5000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    models: Dict[str, Pipeline] = {}
    metrics: List[Dict[str, float]] = []

    # Strategie 1 - niveau algorithmique: class_weight.
    weighted_lr = Pipeline(
        steps=[
            ("preprocess", build_linear_preprocessor(k_best=k_best)),
            (
                "model",
                clone(base_lr).set_params(class_weight="balanced"),
            ),
        ]
    )
    weighted_lr.fit(X_train, y_train)
    proba_weighted = weighted_lr.predict_proba(X_test)[:, 1]
    t_weighted = decision_threshold_from_cost(cost)
    metrics.append(
        evaluate_binary_model(
            "LR ElasticNet - class_weight",
            y_test,
            proba_weighted,
            t_weighted,
            cost,
            out_dir,
        )
    )
    models["LR ElasticNet - class_weight"] = weighted_lr

    # Strategie 2 - niveau donnees: SMOTE.
    smote_lr = ImbPipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=mutual_info_classif, k=k_best)),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("model", clone(base_lr)),
    ]
)

    smote_lr.fit(X_train, y_train)
    proba_smote = smote_lr.predict_proba(X_test)[:, 1]
    t_smote, _ = optimize_threshold_by_mcc(y_test, proba_smote)
    metrics.append(
        evaluate_binary_model(
            "LR ElasticNet - SMOTE",
            y_test,
            proba_smote,
            t_smote,
            cost,
            out_dir,
        )
    )
    models["LR ElasticNet - SMOTE"] = smote_lr

    # Strategie 3 - niveau donnees: ADASYN.
    # Si ADASYN echoue sur certains jeux tres rares, on continue sans arreter le projet.
    try:
        adasyn_lr = ImbPipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=mutual_info_classif, k=k_best)),
        ("adasyn", ADASYN(random_state=RANDOM_STATE, n_neighbors=5)),
        ("model", clone(base_lr)),
    ]
)
        adasyn_lr.fit(X_train, y_train)
        proba_adasyn = adasyn_lr.predict_proba(X_test)[:, 1]
        t_adasyn, _ = optimize_threshold_by_mcc(y_test, proba_adasyn)
        metrics.append(
            evaluate_binary_model(
                "LR ElasticNet - ADASYN",
                y_test,
                proba_adasyn,
                t_adasyn,
                cost,
                out_dir,
            )
        )
        models["LR ElasticNet - ADASYN"] = adasyn_lr
    except Exception as exc:
        print("ADASYN a echoue, raison:", repr(exc))

    return models, metrics


# ---------------------------------------------------------------------------
# 5. Random Forest + matrice de proximite
# ---------------------------------------------------------------------------


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cost: CostConfig,
    out_dir: Path,
) -> Tuple[Pipeline, Dict[str, float]]:
    """Random Forest robuste avec class_weight."""
    rf = Pipeline(
        steps=[
            ("preprocess", build_tree_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    t, _ = optimize_threshold_by_mcc(y_test, proba)
    metrics = evaluate_binary_model("Random Forest - class_weight", y_test, proba, t, cost, out_dir)
    return rf, metrics


def random_forest_proximity_outliers(
    rf_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
    n_sample: int = 800,
) -> pd.DataFrame:
    """Construit une matrice de proximite RF et detecte les points ou le modele hesite.

    La proximite entre deux observations = proportion d'arbres ou les deux observations
    finissent dans la meme feuille terminale.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X_test)
    idx = rng.choice(np.arange(n), size=min(n_sample, n), replace=False)

    X_sub_raw = X_test.iloc[idx].copy()
    y_sub = y_test.iloc[idx].to_numpy()

    preprocessor: Pipeline = rf_pipeline.named_steps["preprocess"]
    rf_model: RandomForestClassifier = rf_pipeline.named_steps["model"]
    X_sub = preprocessor.transform(X_sub_raw)

    # feuilles terminales: shape = (n_observations, n_arbres)
    leaves = rf_model.apply(X_sub)
    n_obs, n_trees = leaves.shape

    proximity = np.zeros((n_obs, n_obs), dtype=np.float32)
    for j in range(n_trees):
        leaf_ids = leaves[:, j]
        proximity += (leaf_ids[:, None] == leaf_ids[None, :]).astype(np.float32)
    proximity /= float(n_trees)

    proba = rf_model.predict_proba(X_sub)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    uncertainty = 1.0 - 2.0 * np.abs(proba - 0.5)
    uncertainty = np.clip(uncertainty, 0, 1)

    # Si un point a peu de proximite avec les points de sa classe predite,
    # il est atypique dans l'espace appris par la foret.
    lack_proximity = []
    for i in range(n_obs):
        same_pred = np.where(y_pred == y_pred[i])[0]
        same_pred = same_pred[same_pred != i]
        if len(same_pred) == 0:
            lack_proximity.append(1.0)
        else:
            lack_proximity.append(1.0 - float(np.mean(proximity[i, same_pred])))
    lack_proximity = np.asarray(lack_proximity)

    error_flag = (y_pred != y_sub).astype(int)
    outlier_score = 0.50 * uncertainty + 0.35 * lack_proximity + 0.15 * error_flag

    outliers = pd.DataFrame(
        {
            "sample_index_in_test": idx,
            "true_class": y_sub,
            "pred_class_threshold_05": y_pred,
            "proba_positive": proba,
            "uncertainty": uncertainty,
            "lack_of_rf_proximity": lack_proximity,
            "prediction_error": error_flag,
            "outlier_score": outlier_score,
        }
    ).sort_values("outlier_score", ascending=False)

    outliers.to_csv(out_dir / "rf_prediction_outliers.csv", index=False)

    # Visualisation 2D de la dissimilarite 1 - proximite.
    dissimilarity = 1.0 - proximity
    coords = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=RANDOM_STATE,
        normalized_stress="auto",
    ).fit_transform(dissimilarity)

    plt.figure(figsize=(8, 6))
    sizes = 30 + 120 * outlier_score
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=proba,
        s=sizes,
        alpha=0.75,
    )
    plt.colorbar(scatter, label="Probabilite classe positive")
    plt.title("Random Forest proximity - outliers de prediction")
    plt.xlabel("MDS 1")
    plt.ylabel("MDS 2")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "rf_prediction_outliers_scatter.png", dpi=180)
    plt.close()

    return outliers


# ---------------------------------------------------------------------------
# 6. XGBoost cost-sensitive + Optuna
# ---------------------------------------------------------------------------


def preprocess_for_xgb(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer, List[str]]:
    imputer = SimpleImputer(strategy="median")
    X_train_np = imputer.fit_transform(X_train)
    X_valid_np = imputer.transform(X_valid)
    X_test_np = imputer.transform(X_test)
    feature_names = list(X_train.columns)
    return X_train_np, X_valid_np, X_test_np, imputer, feature_names


def plot_optuna_history(study: optuna.Study, out_path: Path, title: str) -> None:
    trials = [t for t in study.trials if t.value is not None and t.state.name == "COMPLETE"]
    if not trials:
        return
    values = np.array([t.value for t in trials], dtype=float)
    best = np.maximum.accumulate(values)
    plt.figure(figsize=(8, 5))
    plt.plot(values, marker="o", linestyle="", label="Objective value")
    plt.plot(best, linewidth=2, label="Best so far")
    plt.xlabel("Trial")
    plt.ylabel("Average Precision / AUPRC")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def optimize_xgb_scale_pos_weight(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    out_dir: Path,
    n_trials: int = 30,
) -> Tuple[xgb.XGBClassifier, optuna.Study]:
    """Optuna TPE pour XGBoost avec scale_pos_weight."""
    n_pos = max(1, int(np.sum(y_train == 1)))
    n_neg = max(1, int(np.sum(y_train == 0)))
    base_spw = n_neg / n_pos

    def objective(trial: optuna.Trial) -> float:
        params = {
            # max_depth 3-10: arbres faibles a moyens, evite surapprentissage.
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            # learning_rate faible: boosting plus stable, surtout avec desequilibre.
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.20, log=True),
            # n_estimators: assez grand, early stopping limite automatiquement.
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            # min_child_weight: regularise les feuilles, important avec classe rare.
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            # subsample et colsample: reduisent variance et surapprentissage.
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.00),
            # alpha/lambda: regularisation L1/L2.
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            # gamma: gain minimum pour split, limite splits inutiles.
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            # autour du ratio theorique n_neg/n_pos.
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", max(1.0, base_spw * 0.25), base_spw * 4.0, log=True
            ),
        }

        model = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        proba_valid = model.predict_proba(X_valid)[:, 1]
        return float(average_precision_score(y_valid, proba_valid))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    plot_optuna_history(
        study,
        out_dir / "optuna_history_xgb_scale_pos_weight.png",
        "Optuna convergence - XGBoost scale_pos_weight",
    )

    best_params = study.best_params
    best_model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    best_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return best_model, study


def weighted_logistic_obj_factory(pos_cost: float, neg_cost: float):
    """Cree une fonction de perte customisee pour xgb.train.

    Loss = cout_classe * log-loss binaire.
    Les positifs ont un poids plus fort pour penaliser les faux negatifs.
    """

    def weighted_logistic_obj(predt: np.ndarray, dtrain: xgb.DMatrix):
        y_true = dtrain.get_label()
        p = sigmoid(predt)
        weights = np.where(y_true == 1, pos_cost, neg_cost)
        grad = weights * (p - y_true)
        hess = weights * p * (1.0 - p)
        hess = np.maximum(hess, 1e-6)
        return grad, hess

    return weighted_logistic_obj


def optimize_xgb_custom_loss(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    cost: CostConfig,
    out_dir: Path,
    n_trials: int = 20,
) -> Tuple[xgb.Booster, optuna.Study, Dict]:
    """Optuna TPE pour XGBoost avec loss customisee asymetrique."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    obj = weighted_logistic_obj_factory(pos_cost=cost.fn, neg_cost=cost.fp)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.005, 0.20, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.00),
            "lambda": trial.suggest_float("lambda", 1e-3, 20.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-5, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "tree_method": "hist",
            "seed": RANDOM_STATE,
            "verbosity": 0,
        }
        rounds = trial.suggest_int("num_boost_round", 150, 800)
        booster = xgb.train(params, dtrain, num_boost_round=rounds, obj=obj, verbose_eval=False)
        margin = booster.predict(dvalid, output_margin=True)
        proba_valid = sigmoid(margin)
        return float(average_precision_score(y_valid, proba_valid))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    plot_optuna_history(
        study,
        out_dir / "optuna_history_xgb_custom_loss.png",
        "Optuna convergence - XGBoost custom asymmetric loss",
    )

    best = dict(study.best_params)
    rounds = int(best.pop("num_boost_round"))
    best.update({"tree_method": "hist", "seed": RANDOM_STATE, "verbosity": 0})
    booster = xgb.train(best, dtrain, num_boost_round=rounds, obj=obj, verbose_eval=False)
    return booster, study, {"params": best, "num_boost_round": rounds}


# ---------------------------------------------------------------------------
# 7. Calibration probabiliste
# ---------------------------------------------------------------------------


def calibrate_prefit_model(
    fitted_model,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str = "isotonic",
):
    """Calibre un modele deja entraine.

    Compatible avec nouvelles et anciennes versions de scikit-learn.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    except Exception:
        calibrator = CalibratedClassifierCV(fitted_model, method=method, cv="prefit")
    calibrator.fit(X_calib, y_calib)
    return calibrator


def reliability_diagram_from_predictions(
    y_true: np.ndarray,
    probas: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    for name, proba in probas.items():
        CalibrationDisplay.from_predictions(
            y_true,
            proba,
            n_bins=10,
            strategy="quantile",
            name=name,
            ax=ax,
        )
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# 8. SHAP interpretabilite
# ---------------------------------------------------------------------------


def run_shap_xgb(
    model: xgb.XGBClassifier,
    X_array: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
    n_sample: int = 2000,
) -> None:
    """SHAP pour identifier les variables les plus influentes."""
    n = X_array.shape[0]
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(np.arange(n), size=min(n_sample, n), replace=False)
    X_sample = X_array[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Importance moyenne absolue.
    shap_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": shap_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out_dir / "shap_importance.csv", index=False)


# ---------------------------------------------------------------------------
# 9. Pipeline principal
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["creditcard", "aps"], default="creditcard")
    parser.add_argument("--data_path", default="data/creditcard.csv")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--calib_size", type=float, default=0.20)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)

    if args.dataset == "creditcard":
        X, y, cost = load_creditcard(args.data_path)
    else:
        X, y, cost = load_aps()

    print(f"Dataset: {args.dataset}")
    print(f"Shape X={X.shape}, positives={int(y.sum())}, positive_rate={y.mean():.6f}")
    print(f"Cost config: FP={cost.fp}, FN={cost.fn}")
    print(f"Theoretical cost threshold: {decision_threshold_from_cost(cost):.6f}")

    run_eda(X, y, out_dir)

    # Split 1: test final intouchable.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Split 2: validation pour Optuna et calibration.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_train_full,
        y_train_full,
        test_size=args.calib_size,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )
    X_valid, X_calib, y_valid, y_calib = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print("Split sizes:")
    print("train:", X_train.shape, "valid:", X_valid.shape, "calib:", X_calib.shape, "test:", X_test.shape)

    all_metrics: List[Dict[str, float]] = []

    # A. Desequilibre + baseline Elastic Net.
    imbalance_models, imbalance_metrics = train_imbalance_strategies(
        X_train, y_train, X_test, y_test, cost, out_dir
    )
    all_metrics.extend(imbalance_metrics)
    for name, model in imbalance_models.items():
        joblib.dump(model, out_dir / f"model_{safe_name(name)}.joblib")

    # B. Random Forest + proximite.
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, cost, out_dir)
    all_metrics.append(rf_metrics)
    joblib.dump(rf_model, out_dir / "model_random_forest.joblib")
    outliers = random_forest_proximity_outliers(rf_model, X_test, y_test, out_dir)
    print("\nTop 10 RF prediction outliers:")
    print(outliers.head(10))

    # C. XGBoost preprocessing.
    X_train_xgb, X_valid_xgb, X_test_xgb, xgb_imputer, feature_names = preprocess_for_xgb(
        X_train, X_valid, X_test
    )
    X_calib_xgb = xgb_imputer.transform(X_calib)
    y_train_np = y_train.to_numpy()
    y_valid_np = y_valid.to_numpy()
    y_test_np = y_test.to_numpy()
    y_calib_np = y_calib.to_numpy()

    # D. XGBoost scale_pos_weight + Optuna.
    xgb_spw, study_spw = optimize_xgb_scale_pos_weight(
        X_train_xgb,
        y_train_np,
        X_valid_xgb,
        y_valid_np,
        out_dir,
        n_trials=args.n_trials,
    )
    proba_xgb_spw = xgb_spw.predict_proba(X_test_xgb)[:, 1]
    threshold_xgb, _ = optimize_threshold_by_mcc(y_valid_np, xgb_spw.predict_proba(X_valid_xgb)[:, 1])
    all_metrics.append(
        evaluate_binary_model(
            "XGBoost - scale_pos_weight + Optuna",
            y_test_np,
            proba_xgb_spw,
            threshold_xgb,
            cost,
            out_dir,
        )
    )
    joblib.dump(xgb_spw, out_dir / "model_xgboost_scale_pos_weight.joblib")
    save_json(study_spw.best_params, out_dir / "best_params_xgb_scale_pos_weight.json")

    # E. XGBoost loss customisee + Optuna.
    custom_booster, study_custom, custom_info = optimize_xgb_custom_loss(
        X_train_xgb,
        y_train_np,
        X_valid_xgb,
        y_valid_np,
        cost,
        out_dir,
        n_trials=max(5, args.n_trials // 2),
    )
    dtest = xgb.DMatrix(X_test_xgb, label=y_test_np)
    dvalid = xgb.DMatrix(X_valid_xgb, label=y_valid_np)
    proba_custom_test = sigmoid(custom_booster.predict(dtest, output_margin=True))
    proba_custom_valid = sigmoid(custom_booster.predict(dvalid, output_margin=True))
    threshold_custom, _ = optimize_threshold_by_mcc(y_valid_np, proba_custom_valid)
    all_metrics.append(
        evaluate_binary_model(
            "XGBoost - custom asymmetric loss + Optuna",
            y_test_np,
            proba_custom_test,
            threshold_custom,
            cost,
            out_dir,
        )
    )
    custom_booster.save_model(str(out_dir / "model_xgboost_custom_loss.json"))
    save_json(custom_info, out_dir / "best_params_xgb_custom_loss.json")

    # F. Calibration: on calibre le meilleur XGB sklearn avec sigmoid et isotonic.
    # On compare les diagrammes de fiabilite avant/apres.
    proba_calib_before = xgb_spw.predict_proba(X_calib_xgb)[:, 1]
    method = "isotonic" if len(y_calib_np) >= 1000 else "sigmoid"
    calibrated_xgb = calibrate_prefit_model(xgb_spw, X_calib_xgb, y_calib_np, method=method)
    proba_calibrated_test = calibrated_xgb.predict_proba(X_test_xgb)[:, 1]
    proba_before_test = xgb_spw.predict_proba(X_test_xgb)[:, 1]

    reliability_diagram_from_predictions(
        y_test_np,
        {
            "Before calibration": proba_before_test,
            f"After calibration ({method})": proba_calibrated_test,
        },
        out_dir / "reliability_diagram_xgb_before_after.png",
        "Reliability diagram - XGBoost before/after calibration",
    )

    threshold_cal, _ = optimize_threshold_by_mcc(y_valid_np, xgb_spw.predict_proba(X_valid_xgb)[:, 1])
    all_metrics.append(
        evaluate_binary_model(
            f"XGBoost calibrated - {method}",
            y_test_np,
            proba_calibrated_test,
            threshold_cal,
            cost,
            out_dir,
        )
    )
    joblib.dump(calibrated_xgb, out_dir / f"model_xgboost_calibrated_{method}.joblib")

    # G. SHAP interpretabilite.
    try:
        run_shap_xgb(xgb_spw, X_test_xgb, feature_names, out_dir)
    except Exception as exc:
        print("SHAP a echoue, raison:", repr(exc))

    # H. Tableau final des metriques.
    metrics_df = pd.DataFrame(all_metrics).sort_values(
        ["auprc_average_precision", "mcc", "f1_macro"], ascending=False
    )
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    print("\nMetrics summary:")
    print(metrics_df)

    # Sauvegarde globale.
    joblib.dump(xgb_imputer, out_dir / "xgb_imputer.joblib")
    save_json(
        {
            "dataset": args.dataset,
            "cost_fp": cost.fp,
            "cost_fn": cost.fn,
            "theoretical_cost_threshold": decision_threshold_from_cost(cost),
            "feature_names": feature_names,
        },
        out_dir / "run_config.json",
    )


if __name__ == "__main__":
    main()
