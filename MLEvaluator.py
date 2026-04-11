# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# CLASSE PRINCIPALE
# ============================================================
class MLEvaluator:

    # --------------------------------------------------------
    # BLOC 1 — Constructeur
    # --------------------------------------------------------
    def __init__(self, train_df, test_df):
        """
        Initialise l'évaluateur avec les données train et test.

        Args:
            train_df : DataFrame d'entraînement (après SMOTE)
            test_df  : DataFrame de test (non modifié)
        """
        self.train_df = train_df        # données d'entraînement
        self.test_df = test_df          # données de test
        self.models = {}                # modèles non entraînés
        self.trained_models = {}        # modèles entraînés
        self.results = None             # résultats des évaluations
        self.feature_names = []         # noms des colonnes features
        self.X_train = None             # features train
        self.X_test = None              # features test
        self.y_train = None             # cible train
        self.y_test = None              # cible test
        self.all_predictions = {}  # prédictions de chaque modèle sur le test

    # --------------------------------------------------------
    # BLOC 2 — Préparation des données
    # --------------------------------------------------------
    def prepare_data(self):
        """
        Sépare les features (X) de la cible (y) pour le train et le test.
        Nettoie les données et stocke tout dans la classe.
        """
        print("Préparation des données...")

        # Copie pour ne pas modifier les données originales
        train_df = self.train_df.copy()
        test_df = self.test_df.copy()

        # Vérifier que la colonne cible existe
        target_col = "DIFFICULTY_encoded"
        if target_col not in train_df.columns:
            raise ValueError(f"Colonne cible '{target_col}' introuvable dans le dataset")

        # Séparer X et y
        self.X_train = train_df.drop(columns=[target_col])
        self.y_train = train_df[target_col]
        self.X_test = test_df.drop(columns=[target_col])
        self.y_test = test_df[target_col]

        # Garder uniquement les colonnes numériques
        self.X_train = self.X_train.select_dtypes(include=[np.number])
        self.X_test = self.X_test.select_dtypes(include=[np.number])

        # Remplacer les valeurs manquantes par 0
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)

        # Sauvegarder les noms des features
        self.feature_names = list(self.X_train.columns)

        print(f"Données prêtes !")
        print(f"   Features : {len(self.feature_names)} colonnes")
        print(f"   Train    : {self.X_train.shape[0]} lignes")
        print(f"   Test     : {self.X_test.shape[0]} lignes")
        print(f"   Classes  : {sorted(self.y_train.unique())}")

    # --------------------------------------------------------
    # BLOC 3 — Initialisation des modèles
    # --------------------------------------------------------
    def initialize_models(self):
        """
        Initialise les 10 modèles ML avec leurs configurations.
        Les modèles sont vides — pas encore entraînés.
        """
        print("\nInitialisation des modèles...")

        self.models = {
            # Séparateur à vaste marge
            "SVM": SVC(
                random_state=42,
                probability=True,   # nécessaire pour calculer l'AUC-ROC
                kernel="rbf",       # frontière de décision non linéaire
                C=1.0               # force de régularisation
            ),

            # Régression logistique
            "Logistic_Regression": LogisticRegression(
                random_state=42,
                max_iter=1000,      # nb max d'itérations pour converger
                solver="liblinear"  # algorithme adapté aux petits datasets
            ),

            # Arbre de décision
            "Decision_Tree": DecisionTreeClassifier(
                random_state=42,
                max_depth=10,           # profondeur max de l'arbre
                min_samples_split=5,    # nb min d'exemples pour couper un noeud
                min_samples_leaf=2      # nb min d'exemples dans une feuille
            ),

            # K plus proches voisins
            "KNN": KNeighborsClassifier(
                n_neighbors=5,      # nombre de voisins à considérer
                weights="uniform"   # tous les voisins ont le même poids
            ),

            # Naive Bayes gaussien
            "Naive_Bayes": GaussianNB(),

            # Forêt aléatoire
            "Random_Forest": RandomForestClassifier(
                n_estimators=100,   # nombre d'arbres
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),

            # Variante plus rapide du Random Forest
            "Extra_Trees": ExtraTreesClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),

            # Bagging — combine plusieurs arbres de décision
            "Bagging": BaggingClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=8),
                n_estimators=50,
                random_state=42
            ),

            # Gradient Boosting — arbres construits séquentiellement
            "Gradient_Boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,  # vitesse d'apprentissage
                max_depth=3,
                random_state=42
            ),

            # Réseau de neurones
            "MLP": MLPClassifier(
                hidden_layer_sizes=(100, 50),   # 2 couches cachées
                max_iter=1000,
                random_state=42,
                early_stopping=True,            # arrêt si plus d'amélioration
                validation_fraction=0.1         # 10% du train pour valider
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                eval_metric="mlogloss",  # métrique pour multiclasse
                use_label_encoder=False
            ),
        }

        print(f"{len(self.models)} modèles initialisés")

    # --------------------------------------------------------
    # BLOC 4 — Évaluation d'un modèle
    # --------------------------------------------------------
    def evaluate_model(self, model, model_name):
        """
        Évalue un modèle entraîné et retourne ses métriques de performance.

        Args:
            model      : modèle déjà entraîné
            model_name : nom du modèle (string)

        Returns:
            dictionnaire contenant toutes les métriques
        """
        # Prédictions sur le test
        y_pred = model.predict(self.X_test)

        # Probabilités de prédiction (pour AUC-ROC)
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            # La plupart des modèles ont predict_proba
            y_pred_proba = model.predict_proba(self.X_test)
        elif hasattr(model, "decision_function"):
            # Fallback pour les modèles sans predict_proba
            y_pred_proba = model.decision_function(self.X_test)

        # Calcul des métriques
        accuracy  = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
        f1        = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)

        # AUC-ROC (multiclasse)
        auc_roc = None
        if y_pred_proba is not None:
            try:
                if len(np.unique(self.y_test)) == 2:
                    # Classification binaire
                    auc_roc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                else:
                    # Classification multiclasse (4 classes ici)
                    auc_roc = roc_auc_score(
                        self.y_test,
                        y_pred_proba,
                        multi_class="ovr",      # One vs Rest
                        average="weighted"
                    )
            except Exception as e:
                print(f"AUC-ROC non calculable pour {model_name} : {e}")
                auc_roc = None

        return {
            "Model"    : model_name,
            "Accuracy" : accuracy,
            "Precision": precision,
            "Recall"   : recall,
            "F1_Score" : f1,
            "AUC_ROC"  : auc_roc,
            "y_pred"   : y_pred,
        }

    # --------------------------------------------------------
    # BLOC 5 — Méthode principale
    # --------------------------------------------------------
    def train_and_evaluate_all(self):
        """
        Méthode principale :
        1. Prépare les données
        2. Initialise les modèles
        3. Entraîne chaque modèle
        4. Évalue chaque modèle
        5. Retourne un DataFrame avec tous les résultats
        """
        print("Lancement de l'évaluation des modèles...")
        print("=" * 60)

        # Étape 1 — Préparer les données
        self.prepare_data()

        # Étape 2 — Initialiser les modèles
        self.initialize_models()

        # Étape 3 — Entraîner et évaluer chaque modèle
        all_results = []

        for model_name, model in self.models.items():
            print(f"\nEntraînement de {model_name}...")

            try:
                # Entraînement
                model.fit(self.X_train, self.y_train)

                # Sauvegarde du modèle entraîné
                self.trained_models[model_name] = model

                self.all_predictions[model_name] = model.predict(self.X_test)

                # Évaluation
                result = self.evaluate_model(model, model_name)
                all_results.append(result)

                # Affichage des résultats
                print(f"   Accuracy  : {result['Accuracy']:.4f}")
                print(f"   Precision : {result['Precision']:.4f}")
                print(f"   Recall    : {result['Recall']:.4f}")
                print(f"   F1-Score  : {result['F1_Score']:.4f}")
                if result["AUC_ROC"] is not None:
                    print(f"   AUC-ROC   : {result['AUC_ROC']:.4f}")

            except Exception as e:
                print(f"Erreur avec {model_name} : {e}")
                continue

        # Étape 4 — Stocker les résultats dans un DataFrame
        results_df = pd.DataFrame(all_results)
        results_df = results_df.drop(columns=["y_pred"])
        results_df = results_df.sort_values("F1_Score", ascending=False)
        results_df = results_df.reset_index(drop=True)
        # Sauvegarde automatique
        self.save()

        self.results = results_df

        print("\n" + "=" * 60)
        print("RÉSULTATS FINAUX")
        print("=" * 60)
        print(results_df.to_string(index=False))
        return results_df


    def save(self, filepath="evaluator.pkl"):
        """
        Sauvegarde l'evaluator entier (modèles entraînés + données)
        pour pouvoir le réutiliser dans les notebooks XAI.

        Args:
            filepath : chemin du fichier de sauvegarde
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Evaluator sauvegardé dans '{filepath}'")

    @staticmethod
    def load(filepath="evaluator.pkl"):
        """
        Charge un evaluator précédemment sauvegardé.

        Args:
            filepath : chemin du fichier à charger
        Returns:
            MLEvaluator avec tous les modèles déjà entraînés
        """
        import pickle
        with open(filepath, "rb") as f:
            evaluator = pickle.load(f)
        print(f"Evaluator chargé depuis '{filepath}'")
        print(f"   Modèles disponibles : {list(evaluator.trained_models.keys())}")
        return evaluator

