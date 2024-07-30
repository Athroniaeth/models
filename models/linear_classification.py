from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def accuracy(y_label: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule l'accuracy pour évaluer la qualité du modèle de classification.

    L'accuracy mesure la proportion de prédictions correctes parmi l'ensemble des prédictions.
    Un score d'accuracy de 1.0 indique que toutes les prédictions sont correctes, tandis qu'un score de 0.0
    signifie que toutes les prédictions sont incorrectes.

    :param y_label: Valeurs réelles du dataset (np.ndarray)
    :param y_pred: Valeurs prédites par le modèle (np.ndarray)
    :return: Accuracy (float)
    """
    return np.mean(y_label == y_pred)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Fonction sigmoid pour transformer les logits en probabilités.

    La fonction sigmoid transforme les valeurs d'entrée en valeurs entre 0 et 1 et ce peu importe la valeur d'entrée,
    Cette fonction à peu de chance d'être d'être dans la moyenne (0.5) car elle progresse ou régresse très vite.
    (Voir la fonction dérivée, représentant la pente de la fonction)

    :param x: Valeurs d'entrée (np.ndarray)
    :return: Valeurs transformées par la fonction sigmoid (np.ndarray)
    """
    return 1 / (1 + np.exp(-x))


@dataclass
class LinearClassification:
    bias: float = 0
    lr: float = 0.001
    n_iters: int = 1000
    weights: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        """
        Entraîner le modèle de régression logistique en ajustant les paramètres (poids et biais) pour minimiser l'erreur.

        La méthode de descente de gradient est utilisée pour ajuster les paramètres du modèle en fonction de l'erreur
        entre les valeurs prédites et les valeurs réelles. La méthode de descente de gradient calcule le gradient de
        la fonction de coût par rapport aux paramètres du modèle, puis ajuste les paramètres dans la direction qui
        minimise la fonction de coût.

        :param inputs: Les valeurs d'entrée du dataset (np.ndarray)
        :param labels: Les valeurs cibles du dataset (np.ndarray)
        :return: None
        """
        # Nombre de samples et de features
        n_samples, n_features = inputs.shape

        # Initialise les paramètres du modèle
        self.bias = 0
        self.weights = np.zeros(n_features)

        # Boucle d'entraînement
        for _ in range(self.n_iters):
            # Prédiction du modèle
            y_predicted = sigmoid(self.logits(inputs))

            # Calcul du gradient (Qui correspond à l'impact de chaque paramètre sur la fonction de coût)
            dw = (1 / n_samples) * np.dot(inputs.T, (y_predicted - labels))

            # Calcul du biais (Qui correspond à l'impact du biais sur la fonction de coût)
            db = (1 / n_samples) * np.sum(y_predicted - labels)

            # Mise à jour des paramètres
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def logits(self, inputs: np.ndarray) -> np.ndarray:
        """
        Prédire les logits (sortie brute) pour les inputs donnés

        :param inputs: Les valeurs d'entrée pour lesquelles nous voulons prédire les logits
        :return: Logits prédits (np.ndarray)
        """
        return np.dot(inputs, self.weights) + self.bias

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Prédire les classes pour les inputs donnés

        :param inputs: Les valeurs d'entrée pour lesquelles nous voulons prédire les classes
        :return: Classes prédites (np.ndarray)
        """
        y_predicted = sigmoid(self.logits(inputs))  # Transforme les logits en probabilités
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]  # Transforme les probabilités en classes
        return np.array(y_predicted_cls)


if __name__ == "__main__":
    # Fonction de scikit-learn pour générer un dataset de classification
    inputs, y = datasets.make_classification(
        n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_informative=2, n_redundant=0, random_state=42
    )

    # Diviser le dataset en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, y, test_size=0.2, random_state=42
    )

    # Initialiser le modèle de régression logistique
    classifier = LinearClassification(lr=1, n_iters=1000)

    # Entraîner le modèle de régression logistique
    classifier.fit(X_train, y_train)

    # Prédire les valeurs de y pour les données de test
    predictions = classifier.predict(X_test)

    # Affiche la précision du modèle
    accu = accuracy(y_test, predictions)
    print("Accuracy:", accu)

    # Affiche le graphique de la classification
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, s=10)
    m2 = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, s=10, alpha=0.6)
    x0, x1 = np.meshgrid(
        np.linspace(inputs[:, 0].min(), inputs[:, 0].max(), 100),
        np.linspace(inputs[:, 1].min(), inputs[:, 1].max(), 100)
    )
    x_new = np.c_[x0.ravel(), x1.ravel()]
    y_pred_line = classifier.predict(x_new)
    y_pred_line = y_pred_line.reshape(x0.shape)
    plt.contourf(x0, x1, y_pred_line, alpha=0.3, cmap=cmap)
    plt.show()
