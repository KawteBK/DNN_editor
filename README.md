# NeuroCraft

Bienvenue dans la première version de NeuroCraft.  
Ce projet permet de travailler avec des configurations de réseaux de neurones profonds et de les visualiser de manière interactive.

D’autres fonctionnalités et améliorations seront ajoutées dans les futures versions.

---

## Prérequis

Avant de lancer l’application, assurez-vous d’avoir :

- Python 3.11 installé
- Une connexion Internet (requise pour le téléchargement des datasets)
- Toutes les dépendances installées depuis `requirement.txt`

---

## Installation

### 1. Créer un environnement virtuel

À l’intérieur du dossier `dnn_editor`, créez un environnement virtuel en utilisant Python 3.11 :

`python3.11 -m venv venv`

---

### 2. Activer l’environnement virtuel

Sous Windows :

`venv\Scripts\activate`

Sous Linux / macOS :

`source venv/bin/activate`

---

## Installation des dépendances

Installez toutes les dépendances nécessaires avec la commande :

`pip install -r requirement.txt`

---

## Lancement de l’application

Pour démarrer l’application, exécutez la classe `mainWindow`.

Exemple :

`python mainWindow.py`

---

## Remarques importantes

Si le chargement de l’aperçu du dataset prend un certain temps, c’est normal.  
Le dataset peut être en cours de téléchargement depuis Internet.

La même chose s’applique avant le début de l’entraînement :  
si le dataset n’est pas déjà téléchargé localement, l’application le téléchargera automatiquement.

Pour cette raison, une connexion Internet est nécessaire lors de la première utilisation d’un dataset.

---

## Futures mises à jour

Ceci est seulement la première version du projet.  
D’autres fonctionnalités et améliorations sont prévues pour les prochaines versions.

Profitez-en, et j’espère que ce projet vous sera utile.


--------ENGLISH----------

# NeuroCraft

Welcome to the first version of the NeuroCraft.  
This project allows you to work with and visualize deep neural network configurations in an interactive way.

More features and improvements will be added in future versions.

---

## Requirements

Before running the application, make sure you have:

- Python 3.11 installed
- Internet access (required for dataset downloads)
- All dependencies installed from `requirement.txt`

---

## Installation

### 1. Create a virtual environment

Inside the `dnn_editor` folder, create a virtual environment using Python 3.11:

`python3.11 -m venv venv`

---

2. Activate the virtual environment
   
On Windows:
`venv\Scripts\activate`
On Linux / macOS:
`source venv/bin/activate`

Install Dependencies

Install all required packages using:

`pip install -r requirement.txt`

---
Running the Application

To start the application, run the mainWindow class.

Example:

`python mainWindow.py`

---
Important Notes

If dataset preview loading takes some time, it is normal.
The dataset may be downloading from the internet.
The same applies before training starts:
if the dataset is not already downloaded locally, the application will download it first.
Because of this, an internet connection is required for first-time dataset usage.

---
Future Updates

This is only the first version of the project.
More features and improvements are planned for future releases.

Enjoy, and I hope this project is helpful.

