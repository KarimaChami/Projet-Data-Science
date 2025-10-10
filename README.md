# Projet Data Science – Prédiction du Churn Client

## Description du projet
L’objectif de ce projet est de **prédire le churn client** (désabonnement) à partir des données clients d’une entreprise de télécommunications.  
L’analyse exploratoire (EDA), la préparation des données, l’entraînement des modèles et l’évaluation des performances ont été réalisés avec Python.

---

## Structure du projet
├── Analyse.ipynb # Exploration et visualisation des données
├── Pipeline.py # Préparation, encodage, split, entraînement des modèles
├── test_pipeline.py # Tests unitaires pour valider la cohérence du pipeline
├──test_pipeline.py # Tests unitaires pour valider la cohérence du pipeline
├── README.md # Documentation du projet
├── requirements.txt



---

## Étapes de préparation des données
1. **Nettoyage et encodage :**
   - Suppression des colonnes non pertinentes (`customerID`, `gender`)
   - Encodage des variables catégorielles avec `LabelEncoder`
   - Conversion de `TotalCharges` en numérique et gestion des valeurs manquantes

2. **Division du jeu de données :**
   - 80% pour l’entraînement, 20% pour le test (`train_test_split`)

3. **Normalisation (optionnelle) :**
   - Application de `StandardScaler` pour certaines versions de modèles

---

##  Modèles entraînés
Deux modèles ont été comparés :
- **Régression Logistique**
- **Random Forest**

---


## Modèle retenu pour la mise en production
Le modèle choisi est la **Régression Logistique** car il offre :
- Un bon équilibre entre précision et rappel  
- Une interprétation facile des coefficients  
- Une rapidité d’entraînement et de prédiction  
- Une meilleure stabilité sur les données testées  

Ce modèle est donc le plus adapté pour une mise en production fiable et explicable.

---

## Tests unitaires
Le fichier `test_pipeline.py` vérifie :
- La cohérence des dimensions entre `X` et `y`  
 
=> Les tests unitaires exécutés avec Pytest ont validé la cohérence des dimensions des données après séparation entre X et y.
---

## Exécution du projet

### Installation des dépendances
```bash
pip install -r requirements.txt







Projet réalisé par Karima chami dans le cadre de la formation Simplon – Developpement AI.