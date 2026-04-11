import numpy as np

class XAIEvaluator:
    """
    Classe mère commune à LIME, MC-LIME et SHAP.
    Contient le code partagé : sélection des instances,
    attributs communs, et utilitaires.
    """

    # Modèles retenus pour le XAI
    SELECTED_MODELS  = ['Random_Forest', 'XGBoost', 'Decision_Tree']

    # Consensus uniquement sur les 2 meilleurs
    CONSENSUS_MODELS = ['Random_Forest', 'XGBoost']

    # Ordre du label encoding
    CLASS_NAMES = ['E+P+', 'E+P-', 'E-P+', 'E-P-']

    def __init__(self, evaluator):
        self.evaluator       = evaluator
        self.X_train         = evaluator.X_train
        self.X_test          = evaluator.X_test
        self.y_test          = evaluator.y_test
        self.feature_names   = evaluator.feature_names
        self.all_predictions = evaluator.all_predictions
        self.trained_models  = {
            k: v for k, v in evaluator.trained_models.items()
            if k in self.SELECTED_MODELS
        }
        self.selected_samples = {}

    def select_samples(self):
        """
        Sélectionne les instances représentatives.
        Consensus sur CONSENSUS_MODELS uniquement.
        Méthode commune à LIME, MC-LIME et SHAP.
        """
        print('\nSÉLECTION DES INSTANCES')
        print('=' * 50)

        unique_labels = sorted(self.y_test.unique())
        class_mapping = {
            code: self.CLASS_NAMES[i]
            for i, code in enumerate(unique_labels)
        }
        print(f'Mapping : {class_mapping}')

        for encoded_class, class_name in class_mapping.items():
            class_indices = self.y_test[
                self.y_test == encoded_class
            ].index.tolist()

            correctly_predicted = []
            for idx in class_indices:
                pos = list(self.y_test.index).index(idx)
                if all(
                    preds[pos] == encoded_class
                    for model_name, preds in self.all_predictions.items()
                    if model_name in self.CONSENSUS_MODELS
                ):
                    correctly_predicted.append(pos)

            print(f'\n{class_name} : {len(correctly_predicted)} '
                  f'avec consensus / {len(class_indices)} totales')

            if len(correctly_predicted) >= 4:
                np.random.seed(42)
                choix = np.random.choice(
                    correctly_predicted, 4, replace=False
                ).tolist()
                print('  → 4 sélectionnées ✓')
            elif correctly_predicted:
                choix = correctly_predicted
                print(f'  → {len(choix)} disponible(s) ⚠️')
            else:
                choix = []
                print('  → Aucune ❌')

            self.selected_samples[class_name] = choix

        total = sum(len(v) for v in self.selected_samples.values())
        print(f'\n✓ Total : {total} instances')