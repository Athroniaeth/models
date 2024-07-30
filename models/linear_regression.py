from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def r2_score(y_label: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule le coefficient de détermination (R^2) pour évaluer la qualité du modèle de régression.

    Le score R^2 mesure la proportion de la variance des valeurs réelles qui est expliquée par le modèle.
    Un score R^2 de 1.0 indique que le modèle explique parfaitement les données, tandis qu'un score de 0.0
    signifie que le modèle n'explique pas mieux que la moyenne des valeurs réelles.

    En élevant les erreurs au carré, nous :
    - Rendons les erreurs négatives positives (par exemple, -1² devient 1)
    - Amplifions les grandes erreurs (1² = 1, 2² = 4, 3² = 9...)

    :param y_label: Valeurs réelles du dataset (np.ndarray)
    :param y_pred: Valeurs prédites par le modèle (np.ndarray)
    :return: Score R^2 (float)
    """

    # Calcul de la somme des erreurs au carré (résidus)
    # Cela représente la variance totale non expliquée par le modèle
    # Cela mesure l'erreur totale entre les valeurs prédites et les valeurs réelles
    residual_sum_squares = np.sum((y_label - y_pred) ** 2)

    # Calcul de la somme des carrés totaux (variance totale)
    # Cela représente la variance des valeurs du dataset par rapport à sa moyenne
    # Nous mesurons à partir du dataset, le score maximal que nous pouvons obtenir avec notre modèle
    total_sum_squares = np.sum((y_label - np.mean(y_label)) ** 2)

    # Calcul du score R^2
    # Nous mesurons le % de variance expliquée par notre modèle
    # Remarque: nous faisons "1 - (rss/tss)" car "rss" est une erreur. Plus il est petit, mieux c'est.
    score = 1 - (residual_sum_squares / total_sum_squares)

    return score


def mean_squared_error(y_label: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule l'erreur quadratique moyenne (MSE) pour évaluer la qualité du modèle de régression.

    L'erreur quadratique moyenne (MSE) mesure la moyenne des carrés des erreurs, c'est-à-dire la moyenne des carrés
    des différences entre les valeurs réelles et les valeurs prédites. Plus l'erreur quadratique moyenne est faible,
    meilleur est le modèle.

    En élevant les erreurs au carré, nous :
    - Rendons les erreurs négatives positives (par exemple, -1² devient 1)
    - Amplifions les grandes erreurs (1² = 1, 2² = 4, 3² = 9...)

    :param y_label: Valeurs réelles du dataset (np.ndarray)
    :param y_pred: Valeurs prédites par le modèle (np.ndarray)
    :return: Erreur quadratique moyenne (float)
    """
    return np.mean((y_label - y_pred) ** 2)


@dataclass
class LinearRegression:
    bias: float = 0
    lr: float = 0.001
    n_iters: int = 1000
    weights: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        """
        Entraîner le modèle de régression linéaire en ajustant les paramètres (poids et biais) pour minimiser l'erreur.

        La méthode de descente de gradient est utilisée pour ajuster les paramètres du modèle en fonction de l'erreur
        entre les valeurs prédites et les valeurs réelles. La méthode de descente de gradient calcule le gradient de
        la fonction de coût par rapport aux paramètres du modèle, puis ajuste les paramètres dans la direction qui
        minimise la fonction de coût.

        :param inputs: Les valeurs d'entrée du dataset (np.ndarray)
        :param labels: Les valeurs cibles du dataset (np.ndarray)
        :return: None
        """
        # Nombre de features est toujours égale à 1
        n_samples, n_features = inputs.shape

        # Initialise les paramètres du modèle
        self.bias = 0
        self.weights = np.zeros(n_features)

        # Boucle d'entraînement
        for _ in range(self.n_iters):
            # Prédiction du modèle
            y_predicted = self.predict(inputs)

            # Calcul du gradient (Qui correspond a l'impact de chaque paramètre sur la fonction de coût)
            dw = (1 / n_samples) * np.dot(inputs.T, (y_predicted - labels))

            # Calcul du biais (Qui correspond a l'impact du biais sur la fonction de coût)
            db = (1 / n_samples) * np.sum(y_predicted - labels)

            # Mise à jour des paramètres
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Prédire les valeurs de y_pred pour les inputs donnés

        La methode "np.dot" ne multiplie que si la dernière dimension est identique
        - shape (100,1) * shape (1) = shape (100,),
        - shape (100,1) * shape (2,) Error,
        - shape (100,2) * shape (1,) Error

        :param inputs: Les valeurs d'entrée pour lesquelles nous voulons prédire les valeurs de y
        :return:
        """
        y_predicted = np.dot(inputs, self.weights) + self.bias
        return y_predicted


if __name__ == "__main__":
    # Fonction de scikit-learn pour générer un dataset de régression
    inputs, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    # Diviser le dataset en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, y, test_size=0.2, random_state=42
    )

    # Avoir un dataset de test permet de voir comment le modèle arrive à généraliser sur des données inconnues
    # Cela permet de vérifier si le modèle a appris les données d'entraînement "par coeur" ou s'il a appris des
    # caractéristiques générales du dataset.

    # Note : '1e-2' est équivalent à 1 avec 2 zéros après la VIRGULE (0.01 = x/100)
    # Note : '1e2' est équivalent à 1 avec 2 zéros après le CHIFFRE (100 = x*100)
    # Les fonctions de coût sont très grandes, donc il est nécessaire de les réduire avec le learning rate

    # Initialiser le modèle de régression linéaire
    regressor = LinearRegression(lr=1e-2, n_iters=1000)

    # Entraîner le modèle de régression linéaire
    regressor.fit(X_train, y_train)

    # Prédire les valeurs de y pour les données de test
    predictions = regressor.predict(X_test)

    # Affiche l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    # Affiche le score de coefficient de détermination (R^2)
    accu = r2_score(y_test, predictions)
    print("Accuracy:", accu)

    # Affiche le graphique de la régression linéaire
    # La ligne devrait passer au milieu des points de sorte à minimiser l'écart entre les points et la ligne
    y_pred_line = regressor.predict(inputs)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(inputs, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()
