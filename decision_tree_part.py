import numpy  as np
import pandas as pd 

def label_encoder(feature_vector):
    """
    Encode les catégories/étiquettes des données (en texte) en entier
    """
    unique_labels, encoded_labels = np.unique(feature_vector, return_inverse=True)
    return encoded_labels


def gini_impurity(labels : np.ndarray):
    """
    Calcul l'impureté de Gini pour un ensemble de prédictions.

    L'idée de l'impureté de Gini est la probabilité de se tromper dans une prédiction si on la prenanit au hasard au sein
    d'un échantillon donné.

    Entrées :
        labels : Un tableau de prédiction

    Sortie:
        La valeur d'impureté de Gini (float)
    """
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts/len(labels)
    return 1.-np.sum(probabilities**2)


def split_data( train_data : np.ndarray, labels : np.ndarray, feature_index : int, threshold : float):
    """
    Partitionne un ensemble d'entraînement (et de réponse) par rapport à une valeur de seuil.

    Entrée:
        train_data    : L'ensemble des données d'entraînement
        labels        : Les réponses correspondantes aux données d'entraînement
        feature_index : L'index du l'étiquette par rapport à laquelle on effectue le partitionnement
        threshold     : Le seuil définissant le partitionnement
    """
    left_mask  = train_data[:, feature_index] < threshold
    right_mask = ~left_mask
    return train_data[left_mask], labels[left_mask], train_data[right_mask], labels[right_mask]


def find_best_split(train_data : np.ndarray, labels : np.ndarray):
    """
    Trouve le meilleur partitionnement pour un ensemble de données d'entrée

    Entrées:
        train_data : Les données d'entrée
        labels     : Les prédictions correspondant à chaque entrée de l'entraînement
    
    Sortie:
        La meilleur étiquette (son index en fait) et la valeur de seuil pour son partitionnement
    """
    best_feature, best_threshold, best_gini = None, None, float('inf')
    for feature_index in range(train_data.shape[1]):
        # Récupère toutes les valeurs mesurées par l'étiquette courante (dans la boucle)
        values = np.unique(train_data[:, feature_index])
        # Recherche du seuil optimal pour cette étiquette :
        for threshold in values:
            _, left_labels, _, right_labels = split_data(train_data, labels, feature_index, threshold)
            if len(left_labels) == 0 or len(right_labels) == 0: continue
            # Calcul de l'indice de gini:
            gini = (len(left_labels)*gini_impurity(left_labels) + len(right_labels)*gini_impurity(right_labels)) / len(labels)
            # Plus l'indice de gini est petit, mieux c'est
            if gini < best_gini:
                best_feature, best_threshold, best_gini = feature_index, threshold, gini
    return best_feature, best_threshold


def build_tree(train_data : np.ndarray, labels : np.ndarray, max_depth : int, min_samples_split : int, depth : int =  0):
    """
    Construit un arbre de décision récursivement

    Entrées :
        train_data        : Les données d'entrée
        labels            : Les prédictions correspondant à chaque entrée de l'entraînement
        max_depth         : La profondeur maximale de l'arbre
        min_samples_split : Le nombre d'échantillons minimal requis pour partitionner un noeud de l'arbre
        depth (optionnel) : La profondeur de l'arbre actuellement traitée (dans la récursion)

    Sortie:
        L'arbre de décision retourné sous la forme d'un dictionnaire
    """
    if depth == max_depth or len(labels) < min_samples_split or gini_impurity(labels) == 0:
        return {'prediction' : np.argmax(np.bincount(labels))}
    feature, threshold = find_best_split(train_data, labels)
    # Si pas trouvé de feature adéquat, on arrête la construction de la branche et
    # on recherche la valeur la plus probable comme réponse (sous forme d'indice) :
    if feature is None: return {"prediction" : np.argmax(np.bincount(labels))}
    left_train_data, left_labels, right_train_data, right_labels = split_data(train_data, labels, feature, threshold)
    return {
        "feature"   : feature,
        "threshold" : threshold,
        "left"      : build_tree(left_train_data, left_labels, max_depth, min_samples_split, depth+1),
        "right"     : build_tree(right_train_data, right_labels, max_depth, min_samples_split, depth+1)
    }

