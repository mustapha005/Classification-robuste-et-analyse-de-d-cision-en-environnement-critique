# Projet final AI — Détection de fraude par carte bancaire

## 1. Présentation générale du projet

Ce projet est un travail final d'apprentissage automatique supervisé sur un problème réel de **classification binaire très déséquilibrée** : la détection de transactions frauduleuses par carte bancaire.

Le but n'est pas seulement d'entraîner un modèle qui prédit bien, mais aussi de respecter une démarche complète de projet AI :

- comprendre et préparer les données ;
- analyser le déséquilibre extrême entre les classes ;
- tester plusieurs familles de modèles ;
- optimiser les hyperparamètres avec Optuna ;
- comparer les modèles avec des métriques adaptées ;
- étudier la calibration des probabilités ;
- interpréter les résultats ;
- expliquer les erreurs et les points où le modèle hésite ;
- produire un rapport final clair et reproductible.

Le projet utilise le dataset **Credit Card Fraud Detection** de Kaggle, recommandé pour ce type de problème de fraude. La variable cible est `Class` :

- `Class = 0` : transaction normale ;
- `Class = 1` : transaction frauduleuse.

---

## 2. Résumé des données

D'après l'exécution finale du script :

| Élément | Valeur |
|---|---:|
| Nombre total d'observations | 284807 |
| Nombre de variables explicatives | 33 |
| Nombre de transactions normales | 284315 |
| Nombre de transactions frauduleuses | 492 |
| Taux de fraude | 0.001727 |
| Taux moyen de valeurs manquantes | 0.000000 |
| Taux maximum de valeurs manquantes | 0.000000 |

Le dataset est donc **extrêmement déséquilibré** : les fraudes représentent environ **0.17 %** des transactions.  
C'est pour cela que l'accuracy n'est pas une métrique fiable ici. Un modèle qui prédit toujours "non fraude" aurait une très grande accuracy, mais serait inutile pour détecter les fraudes.

---

## 3. Structure du projet

La structure recommandée du projet est la suivante :

```text
projet_ai_final/
│
├── data/
│   └── creditcard.csv
│
├── outputs/
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── vif.csv
│   ├── metrics_summary.csv
│   ├── rf_prediction_outliers.csv
│   ├── rf_prediction_outliers_scatter.png
│   ├── optuna_history_xgb_scale_pos_weight.png
│   ├── optuna_history_xgb_custom_loss.png
│   ├── reliability_diagram_xgb_before_after.png
│   ├── pr_curve_*.png
│   └── model_*.joblib / model_*.json
│
├── src/
│   └── project_ai_final.py
│
├── requirements.txt
├── README.md
└── Rapport_Final_Projet_AI_Fraude_FINAL_corrige.docx
```

### Rôle des dossiers

- `data/` contient le dataset original.
- `src/` contient le code Python principal.
- `outputs/` contient les résultats générés automatiquement : figures, tableaux, modèles sauvegardés et métriques.
- `requirements.txt` contient les bibliothèques Python nécessaires.
- `Rapport_Final_Projet_AI_Fraude_FINAL_corrige.docx` est le rapport final écrit.

---

## 4. Installation locale

### 4.1 Créer un environnement virtuel

Dans le dossier du projet, exécuter :

```bash
python -m venv .venv
```

Sur Windows avec Git Bash :

```bash
source .venv/Scripts/activate
```

Sur Linux ou macOS :

```bash
source .venv/bin/activate
```

Quand l'environnement est activé, le terminal affiche normalement :

```text
(.venv)
```

---

### 4.2 Installer les dépendances

```bash
pip install -r requirements.txt
```

Si l'installation est lente ou bloque à cause de la connexion internet, utiliser :

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --default-timeout=1000 --retries 10
```

Si `shap`, `numba` ou `llvmlite` posent problème, installer d'abord les bibliothèques essentielles :

```bash
pip install numpy pandas scikit-learn matplotlib seaborn statsmodels imbalanced-learn xgboost optuna joblib --default-timeout=1000 --retries 10
```

Puis installer SHAP séparément :

```bash
pip install shap --default-timeout=1000 --retries 10
```

---

## 5. Télécharger le dataset

Le dataset utilisé est :

```text
Credit Card Fraud Detection
```

Page Kaggle :

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

Après téléchargement, placer le fichier ici :

```text
projet_ai_final/data/creditcard.csv
```

Le chemin final doit être :

```text
data/creditcard.csv
```

---

## 6. Exécuter le projet

### Exécution rapide pour vérifier que tout fonctionne

```bash
python src/project_ai_final.py --dataset creditcard --data_path data/creditcard.csv --n_trials 5
```

Cette commande lance seulement 5 essais Optuna. Elle est utile pour tester le script rapidement.

### Exécution finale utilisée pour le rapport

```bash
python src/project_ai_final.py --dataset creditcard --data_path data/creditcard.csv --n_trials 30
```

Cette commande lance 30 essais Optuna pour mieux optimiser les hyperparamètres XGBoost.

---

## 7. Différence entre `--n_trials 5` et `--n_trials 30`

Le paramètre `--n_trials` contrôle le nombre d'essais effectués par Optuna.

- `--n_trials 5` : rapide, utile pour tester le code.
- `--n_trials 30` : plus long, mais plus sérieux pour le rapport final.
- Plus il y a d'essais, plus Optuna a de chances de trouver une bonne combinaison d'hyperparamètres.

Dans le rapport final, l'exécution utilisée est :

```bash
python src/project_ai_final.py --dataset creditcard --data_path data/creditcard.csv --n_trials 30
```

---

## 8. Méthodologie générale

Le projet suit cette méthode :

1. Chargement des données.
2. Préparation des variables.
3. Analyse exploratoire des données.
4. Analyse du déséquilibre des classes.
5. Séparation des données en train, validation, calibration et test.
6. Utilisation de splits stratifiés pour conserver le même taux de fraude dans chaque partie.
7. Entraînement de plusieurs modèles.
8. Optimisation des hyperparamètres avec Optuna.
9. Évaluation avec des métriques adaptées aux classes déséquilibrées.
10. Calibration des probabilités.
11. Analyse des points difficiles pour le Random Forest.
12. Interprétation et conclusion.

---

## 9. Préparation des données

Le dataset contient les variables `V1` à `V28`, qui sont déjà transformées par PCA dans le dataset original.  
Le script ajoute aussi des variables utiles :

- `Hour` : heure approximative de la transaction à partir de `Time` ;
- `LogAmount` : transformation logarithmique du montant ;
- `Amount_to_median` : rapport entre le montant et la médiane.

Ces variables aident à mieux représenter le comportement temporel et monétaire des transactions.

---

## 10. Analyse exploratoire des données

### 10.1 Distribution des classes

Le graphique `class_distribution.png` montre que la classe 0 domine presque totalement la classe 1.  
Cela confirme que le problème est fortement déséquilibré.

Conséquence : il ne faut pas utiliser l'accuracy comme métrique principale.

### 10.2 Matrice de corrélation

Le graphique `correlation_matrix.png` montre les variables les plus liées à la cible.  
Certaines variables PCA comme `V14`, `V17`, `V12` ou `V10` apparaissent importantes pour séparer les fraudes des transactions normales.

### 10.3 VIF

Le fichier `vif.csv` mesure la multicolinéarité entre variables.

Les variables avec les VIF les plus élevés sont :

```text
Time, Hour, LogAmount
```

Cela est logique car `Hour` est dérivée de `Time`, et `LogAmount` est dérivée de `Amount`.  
Les autres variables ont des VIF faibles, ce qui indique peu de multicolinéarité problématique.

---

## 11. Validation croisée stratifiée

Le projet utilise une logique de séparation stratifiée et de validation adaptée au déséquilibre.

La stratification est importante parce que la classe fraude est très rare.  
Sans stratification, certains sous-ensembles pourraient contenir trop peu de fraudes ou même aucune fraude.

Dans le code, l'idée est équivalente à :

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

`StratifiedKFold` conserve approximativement la même proportion de classes dans chaque fold.  
Cela rend l'évaluation plus stable et plus fiable pour un dataset avec seulement 492 fraudes sur 284807 transactions.

---

## 12. Gestion du déséquilibre des classes

Le projet compare plusieurs stratégies.

### 12.1 `class_weight`

Cette méthode donne plus de poids à la classe minoritaire pendant l'apprentissage.  
Elle ne modifie pas les données, mais force le modèle à prendre les fraudes plus au sérieux.

### 12.2 SMOTE

SMOTE crée de nouveaux exemples synthétiques de la classe minoritaire.  
Cela permet au modèle d'avoir plus d'exemples de fraude pendant l'entraînement.

### 12.3 ADASYN

ADASYN ressemble à SMOTE, mais génère davantage d'exemples synthétiques dans les zones difficiles où la classe minoritaire est mal séparée.

### 12.4 `scale_pos_weight`

Dans XGBoost, `scale_pos_weight` augmente le poids des fraudes dans la fonction de perte.  
C'est une méthode très utile lorsque la classe positive est rare.

### 12.5 Fonction de perte asymétrique

Le projet utilise aussi une perte asymétrique personnalisée pour donner plus d'importance aux faux négatifs.  
Dans un problème de fraude, rater une fraude est souvent plus coûteux que signaler une transaction normale comme suspecte.

Dans ce projet :

```text
coût FP = 1.0
coût FN = 50.0
seuil théorique = 0.019608
```

Cela signifie qu'un faux négatif est considéré comme 50 fois plus grave qu'un faux positif.

---

## 13. Modèles entraînés

### 13.1 Logistic Regression Elastic Net

La régression logistique Elastic Net est un modèle linéaire régularisé.  
Elle combine deux pénalisations :

- L1 : favorise la sélection de variables ;
- L2 : stabilise les coefficients.

Elle est utile comme modèle de référence parce qu'elle est plus interprétable qu'un modèle complexe.

Trois versions sont testées :

- Elastic Net avec `class_weight` ;
- Elastic Net avec SMOTE ;
- Elastic Net avec ADASYN.

---

### 13.2 Random Forest avec `class_weight`

Le Random Forest est un ensemble d'arbres de décision.  
Il permet de capturer des relations non linéaires entre les variables.

Dans ce projet, il est aussi utilisé pour analyser les proximités entre observations.  
Deux observations sont considérées proches si elles tombent souvent dans les mêmes feuilles des arbres.

Cela permet d'identifier des transactions difficiles ou atypiques :

- probabilité proche de la zone d'hésitation ;
- faible proximité avec les autres observations ;
- erreur de prédiction ;
- score d'outlier élevé.

---

### 13.3 XGBoost avec `scale_pos_weight`

XGBoost est un modèle de gradient boosting très performant.  
Il construit des arbres successifs, où chaque arbre corrige les erreurs du précédent.

La version `scale_pos_weight` traite directement le déséquilibre en donnant plus de poids à la classe fraude.

---

### 13.4 XGBoost avec fonction de perte personnalisée

Cette version utilise une perte asymétrique pour pénaliser davantage les fraudes ratées.  
Elle est adaptée au contexte métier où manquer une fraude coûte plus cher que faire une fausse alerte.

---

### 13.5 XGBoost calibré isotonic

Après l'entraînement, les probabilités XGBoost sont calibrées avec une méthode isotonic.  
La calibration sert à rendre les probabilités plus fiables.

Par exemple, si le modèle prédit 0.80, cela devrait correspondre à environ 80 % de vrais positifs dans ce groupe.

---

## 14. Justification théorique des principaux hyperparamètres

### 14.1 `max_depth`

`max_depth` contrôle la profondeur maximale des arbres.

- Une profondeur faible réduit le risque d'overfitting.
- Une profondeur élevée permet de capturer des interactions complexes.
- Dans ce projet, la recherche utilise des valeurs modérées pour garder un bon compromis entre biais et variance.

### 14.2 `learning_rate` / `eta`

Le learning rate contrôle la contribution de chaque arbre.

- Une petite valeur apprend plus lentement mais plus prudemment.
- Une valeur trop grande peut rendre l'entraînement instable.
- Optuna cherche une valeur qui permet d'apprendre efficacement sans sur-ajuster.

### 14.3 `n_estimators` / `num_boost_round`

Ce paramètre représente le nombre d'arbres.

- Plus d'arbres peuvent améliorer la performance.
- Trop d'arbres peuvent augmenter le temps de calcul et l'overfitting.
- La valeur optimale dépend du learning rate.

### 14.4 `min_child_weight`

Ce paramètre impose une quantité minimale d'information dans une feuille.

- Une valeur élevée rend le modèle plus conservateur.
- Une valeur faible permet de créer des feuilles plus spécifiques.
- Il est utile pour éviter que le modèle apprenne du bruit dans un dataset déséquilibré.

### 14.5 `subsample`

`subsample` contrôle la proportion des lignes utilisées pour chaque arbre.

- Une valeur inférieure à 1 ajoute de l'aléatoire.
- Cela réduit l'overfitting.
- Cela rend le modèle plus robuste.

### 14.6 `colsample_bytree`

Ce paramètre contrôle la proportion de variables utilisées pour chaque arbre.

- Il réduit la dépendance à quelques variables.
- Il améliore la généralisation.
- Il est utile quand plusieurs variables sont corrélées ou redondantes.

### 14.7 `reg_alpha`

`reg_alpha` correspond à la régularisation L1.

- Elle pousse certains poids vers zéro.
- Elle favorise des modèles plus simples.
- Elle peut aider quand beaucoup de variables sont peu utiles.

### 14.8 `reg_lambda` / `lambda`

`reg_lambda` correspond à la régularisation L2.

- Elle limite les poids trop grands.
- Elle stabilise le modèle.
- Elle réduit l'overfitting.

### 14.9 `gamma`

`gamma` contrôle le gain minimum nécessaire pour faire un split.

- Une valeur élevée rend les arbres plus prudents.
- Elle évite les splits inutiles.
- Elle améliore la généralisation.

### 14.10 `scale_pos_weight`

Ce paramètre augmente l'importance de la classe positive.

- Il est essentiel lorsque les fraudes sont rares.
- Il force XGBoost à réduire les faux négatifs.
- Optuna l'optimise pour trouver le meilleur équilibre entre rappel, précision et coût métier.

---

## 15. Meilleurs hyperparamètres trouvés

### 15.1 XGBoost `scale_pos_weight`

```json
{
  "max_depth": 10,
  "learning_rate": 0.1252125943833924,
  "n_estimators": 519,
  "min_child_weight": 3.986511459554404,
  "subsample": 0.7163801406750037,
  "colsample_bytree": 0.7833121885702731,
  "reg_alpha": 0.0006131191474419425,
  "reg_lambda": 0.008433579358582122,
  "gamma": 0.8853156768320467,
  "scale_pos_weight": 1946.822282410901
}
```

### 15.2 XGBoost custom asymmetric loss

```json
{
  "params": {
    "max_depth": 3,
    "eta": 0.08888628409181555,
    "min_child_weight": 1.6986483836707045,
    "subsample": 0.7737849726003561,
    "colsample_bytree": 0.9142810499193087,
    "lambda": 0.0013026066818627973,
    "alpha": 7.295272712759666e-05,
    "gamma": 1.5135202009605735,
    "tree_method": "hist",
    "seed": 42,
    "verbosity": 0
  },
  "num_boost_round": 767
}
```

---

## 16. Résultats principaux

Les métriques finales sont les suivantes :

| model                                     |   threshold |   f1_macro |   auprc_average_precision |      mcc |   brier_loss |   log_loss |   business_cost |
|:------------------------------------------|------------:|-----------:|--------------------------:|---------:|-------------:|-----------:|----------------:|
| XGBoost - scale_pos_weight + Optuna       |    0.001205 |   0.615120 |                  0.871590 | 0.346517 |     0.000431 |   0.002893 |     1062.000000 |
| XGBoost - custom asymmetric loss + Optuna |    0.002613 |   0.622487 |                  0.866267 | 0.358749 |     0.000426 |   0.002873 |     1019.000000 |
| Random Forest - class_weight              |    0.047895 |   0.631070 |                  0.851161 | 0.374615 |     0.000590 |   0.005383 |      931.000000 |
| XGBoost calibrated - isotonic             |    0.001205 |   0.555865 |                  0.836466 | 0.241476 |     0.000383 |   0.003146 |     1710.000000 |
| LR ElasticNet - SMOTE                     |    0.865885 |   0.629564 |                  0.722986 | 0.370359 |     0.025331 |   0.115546 |      982.000000 |
| LR ElasticNet - class_weight              |    0.019608 |   0.323909 |                  0.709066 | 0.038348 |     0.024031 |   0.112992 |    30068.000000 |
| LR ElasticNet - ADASYN                    |    0.988786 |   0.628059 |                  0.644785 | 0.366103 |     0.065866 |   0.265553 |     1033.000000 |

---

## 17. Interprétation des métriques

### 17.1 F1-Macro

Le F1-Macro calcule le F1-score séparément pour chaque classe, puis fait la moyenne.  
Il est utile quand les classes sont déséquilibrées car il donne de l'importance à la classe minoritaire.

### 17.2 AUPRC

AUPRC signifie Area Under the Precision-Recall Curve.  
Cette métrique est très importante pour les problèmes de fraude parce qu'elle mesure le compromis entre précision et rappel sur la classe positive.

### 17.3 MCC

MCC signifie Matthews Correlation Coefficient.  
C'est une métrique robuste pour les classifications déséquilibrées, car elle utilise les quatre valeurs de la matrice de confusion.

### 17.4 Brier Loss

Le Brier Loss mesure la qualité des probabilités prédites.  
Plus il est faible, plus les probabilités sont fiables.

### 17.5 Log Loss

Le Log Loss pénalise les mauvaises probabilités, surtout quand le modèle est très confiant mais faux.

### 17.6 Business Cost

Le coût métier est calculé avec :

```text
coût = FP * 1 + FN * 50
```

Ce coût reflète le fait que manquer une fraude est beaucoup plus grave que produire une fausse alerte.

---

## 18. Analyse des résultats

### 18.1 Meilleur modèle selon le coût métier

Le modèle avec le plus faible coût métier est :

```text
Random Forest - class_weight
```

Son coût métier est :

```text
931
```

Il obtient aussi le meilleur MCC parmi les modèles testés :

```text
MCC = 0.374615
```

Cela indique qu'il trouve un bon équilibre entre détection des fraudes et limitation des fausses alertes.

### 18.2 Meilleur modèle selon AUPRC

Le meilleur AUPRC est obtenu par :

```text
XGBoost - scale_pos_weight + Optuna
```

avec :

```text
AUPRC = 0.871590
```

Cela signifie que XGBoost est très performant pour classer les transactions selon leur risque de fraude.

### 18.3 Meilleur modèle global

Même si XGBoost a la meilleure AUPRC, le Random Forest donne le coût métier le plus faible dans ce run.  
Pour une application réelle, on peut choisir :

- Random Forest si l'objectif principal est de minimiser le coût métier ;
- XGBoost si l'objectif principal est d'obtenir le meilleur classement des transactions à risque ;
- XGBoost calibré si l'objectif principal est d'obtenir des probabilités plus fiables.

---

## 19. Analyse Random Forest proximity / outliers

Le fichier `rf_prediction_outliers.csv` contient les transactions les plus difficiles pour le Random Forest.

Ces points sont considérés comme difficiles quand :

- le modèle donne une probabilité ambiguë ;
- la transaction est peu proche des autres transactions dans l'espace des arbres ;
- le modèle se trompe ;
- le score d'outlier est élevé.

Dans les premiers outliers, plusieurs transactions sont de vraie classe 0 mais reçoivent une probabilité positive non négligeable.  
Cela signifie que leurs caractéristiques ressemblent partiellement à des fraudes, même si elles ne sont pas frauduleuses.

Exemple typique :

```text
true_class = 0
proba_positive élevée
lack_of_rf_proximity proche de 1
```

Interprétation :

- la transaction normale est atypique ;
- elle tombe dans des feuilles rarement partagées avec les autres transactions ;
- elle peut contenir des valeurs de variables proches de transactions frauduleuses ;
- le modèle hésite parce qu'elle ne ressemble pas fortement aux transactions normales classiques.

Un autre cas important est une transaction normale prédite comme fraude à seuil 0.5.  
C'est un faux positif : le modèle préfère signaler une transaction suspecte plutôt que de prendre le risque de manquer une fraude.

Cette analyse répond à la question : pourquoi le modèle échoue ou hésite ?  
Il hésite parce que certains points sont isolés dans l'espace de proximité Random Forest et ont des profils statistiques proches de la classe fraude.

---

## 20. Optuna et convergence

Optuna est utilisé pour optimiser les hyperparamètres de XGBoost.

Les figures :

```text
optuna_history_xgb_scale_pos_weight.png
optuna_history_xgb_custom_loss.png
```

montrent l'évolution de la valeur objectif au fil des essais.

La courbe "Best so far" augmente par paliers.  
Cela signifie qu'Optuna teste plusieurs combinaisons et conserve progressivement les meilleures.

Dans le rapport, cela permet de montrer que l'optimisation n'est pas aléatoire seulement : elle explore plusieurs zones de l'espace d'hyperparamètres et améliore progressivement le modèle.

---

## 21. Courbes Precision-Recall

Les courbes Precision-Recall sont sauvegardées pour chaque modèle :

```text
pr_curve_lr_elasticnet___class_weight.png
pr_curve_lr_elasticnet___smote.png
pr_curve_lr_elasticnet___adasyn.png
pr_curve_random_forest___class_weight.png
pr_curve_xgboost___scale_pos_weight_+_optuna.png
pr_curve_xgboost___custom_asymmetric_loss_+_optuna.png
pr_curve_xgboost_calibrated___isotonic.png
```

Ces courbes sont plus adaptées que les courbes ROC dans ce projet, car la classe fraude est très rare.

Une bonne courbe Precision-Recall reste proche du coin supérieur droit :

- précision élevée ;
- rappel élevé.

---

## 22. Calibration des probabilités

La calibration est étudiée avec :

```text
reliability_diagram_xgb_before_after.png
```

Le diagramme compare :

- les probabilités avant calibration ;
- les probabilités après calibration isotonic ;
- la ligne parfaite de calibration.

La calibration ne cherche pas forcément à améliorer le F1 ou l'AUPRC.  
Son objectif est de rendre les probabilités plus interprétables et plus fiables.

Dans ce projet, le modèle calibré obtient un Brier Loss plus faible :

```text
XGBoost calibré isotonic : Brier Loss = 0.000383
```

Cela indique de meilleures probabilités, même si le coût métier augmente.

---

## 23. Fichiers importants générés

### 23.1 Fichiers de résultats

| Fichier | Rôle |
|---|---|
| `metrics_summary.csv` | Tableau final des métriques |
| `eda_summary.json` | Résumé statistique du dataset |
| `vif.csv` | Analyse de multicolinéarité |
| `rf_prediction_outliers.csv` | Points difficiles du Random Forest |
| `run_config.json` | Configuration finale du run |
| `best_params_xgb_scale_pos_weight.json` | Meilleurs hyperparamètres XGBoost avec scale_pos_weight |
| `best_params_xgb_custom_loss.json` | Meilleurs hyperparamètres XGBoost custom loss |

### 23.2 Figures

| Figure | Rôle |
|---|---|
| `class_distribution.png` | Montre le déséquilibre des classes |
| `correlation_matrix.png` | Analyse des corrélations |
| `rf_prediction_outliers_scatter.png` | Visualisation des outliers Random Forest |
| `optuna_history_xgb_scale_pos_weight.png` | Convergence Optuna pour XGBoost pondéré |
| `optuna_history_xgb_custom_loss.png` | Convergence Optuna pour XGBoost custom loss |
| `reliability_diagram_xgb_before_after.png` | Calibration avant/après |
| `pr_curve_*.png` | Courbes Precision-Recall |

### 23.3 Modèles sauvegardés

| Modèle sauvegardé | Description |
|---|---|
| `model_lr_elasticnet___class_weight.joblib` | Logistic Regression Elastic Net pondérée |
| `model_lr_elasticnet___smote.joblib` | Logistic Regression avec SMOTE |
| `model_lr_elasticnet___adasyn.joblib` | Logistic Regression avec ADASYN |
| `model_random_forest.joblib` | Random Forest |
| `model_xgboost_scale_pos_weight.joblib` | XGBoost pondéré |
| `model_xgboost_custom_loss.json` | XGBoost avec perte custom |
| `model_xgboost_calibrated_isotonic.joblib` | XGBoost calibré |

---

## 24. Commandes utiles

### Relancer tout le projet

```bash
python src/project_ai_final.py --dataset creditcard --data_path data/creditcard.csv --n_trials 30
```

### Relancer rapidement

```bash
python src/project_ai_final.py --dataset creditcard --data_path data/creditcard.csv --n_trials 5
```

### Lire les métriques finales dans Python

```python
import pandas as pd

metrics = pd.read_csv("outputs/metrics_summary.csv")
print(metrics)
```

### Voir les meilleurs outliers RF

```python
import pandas as pd

outliers = pd.read_csv("outputs/rf_prediction_outliers.csv")
print(outliers.head(10))
```

### Lire les meilleurs hyperparamètres XGBoost

```python
import json

with open("outputs/best_params_xgb_scale_pos_weight.json") as f:
    params = json.load(f)

print(params)
```

---

## 25. Problèmes fréquents et solutions

### 25.1 `ModuleNotFoundError: No module named 'joblib'`

Solution :

```bash
pip install -r requirements.txt
```

Ou seulement :

```bash
pip install joblib
```

---

### 25.2 Kaggle demande `username`

Cela signifie que le fichier `kaggle.json` n'est pas configuré correctement.

Créer :

```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
```

Puis mettre :

```json
{
  "username": "VOTRE_USERNAME_KAGGLE",
  "key": "VOTRE_CLE_API"
}
```

Ensuite :

```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

### 25.3 Le téléchargement Kaggle est trop lent

Solution simple :

1. Télécharger manuellement le dataset depuis Kaggle.
2. Extraire le ZIP.
3. Placer `creditcard.csv` dans `data/`.

---

### 25.4 Erreur avec `--sample_size`

Si le script affiche :

```text
unrecognized arguments: --sample_size
```

Il faut supprimer ce paramètre et lancer :

```bash
python src/project_ai_final.py --dataset creditcard --data_path data/creditcard.csv --n_trials 5
```

---

### 25.5 Activation de venv sur Windows Git Bash

Utiliser :

```bash
source .venv/Scripts/activate
```

et non :

```bash
source .venv/bin/activate
```

---

## 26. Conclusion courte

Ce projet montre que la détection de fraude nécessite une méthodologie différente d'une classification classique.  
À cause du déséquilibre extrême, l'accuracy est insuffisante.  
Les métriques les plus importantes sont F1-Macro, AUPRC, MCC et le coût métier.

Le Random Forest donne le meilleur coût métier dans l'exécution finale, tandis que XGBoost donne la meilleure AUPRC.  
La calibration isotonic améliore la qualité des probabilités, mais n'améliore pas forcément le coût métier.

Le projet respecte donc les exigences principales :

- EDA complète ;
- gestion du déséquilibre ;
- modèles linéaires, Random Forest et XGBoost ;
- optimisation Optuna ;
- justification des hyperparamètres ;
- validation stratifiée ;
- métriques avancées ;
- calibration ;
- analyse des erreurs et des outliers ;
- rapport final reproductible.

---

## 27. Auteur / contexte

Projet réalisé dans le cadre du projet final AI/ML par : Mustapha Aarab , Aya Agrigah , Atiqa Essayouti.  
Sujet : détection de fraude bancaire avec données fortement déséquilibrées.

