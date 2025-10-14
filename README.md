## Naturun_The_Goat

Projet de deuxième année de Master MIAGE à l'Université Evry-Paris Saclay en Projet Applicatif/Architecture Orientée Services

##  Sommaire
- [Présentation du repository](#-présentation-du-repository)
- [Présentation de l’équipe](#-présentation-de-léquipe--les-rôles-de-chacun)
- [Présentation du projet](#-présentation-du-projet)
- [Ambition finale](#-ambition-finale)
- [Règles de développement](#-règles-de-développement)
- [Installation & utilisation](#️-installation--utilisation)

# 🧩 Projet Naturun
### 📘 Présentation du repository

Ce dépôt contient le Back Office de l’application, développé principalement avec Python.

### 👥 Présentation de l’équipe & les rôles de chacun

- Aymerick RAKOTOARIVONY : Lead Front
- Yohann FREMONT : Coordinateur Front
- Lilian MARIE-JOSEPH : DevOps Docker
- Davidson CHARLOT : Dev IA
- Nathan BAPIN : Lead Dev Back
- Quentin AYRAL : Coordinateur Back
- Idel SADI : Dev Full stack

### 🚀 Présentation du projet

Ce projet a pour but de créer une IA capable de réussir un labyrinthe avec le moins de coups possibles.
Un des objectifs est de réaliser plusieurs difficultés dans un premier temps puis de permettre d'y jouer de manière interactive.

### 🎯 Ambition finale

- Créer une IA performante.
- Apprendre à travailler en groupe et synchroniser le travail de deux équipes.
- Accroître les connaissances dans des domaines nouveaux (Intelligence artificielle, TypeScript, Docker, gestion d'API).

### 🧱 Règles de développement

#### 🪣 Gestion des issues

- Crée une issue pour chaque tâche / bug.
- Utilise les labels (bug, feature, enhancement, documentation, etc.).
- Lie chaque issue à une PR.

#### 🌿 Nommage des branches

- feature/<nom_fonctionnalité> pour les nouvelles features.
- fix/<nom_bug> pour les corrections.
- refactor/<nom> pour les refactorisations.

#### 🔁 Pull Requests

Une PR = une fonctionnalité / un correctif.
Vérifie que les tests passent avant soumission.
Au moins 1 review approuvée avant merge. (aide de GitHub Copilot)

#### ✅ Tests & couverture

Tous les nouveaux modules doivent avoir des tests unitaires.
Couverture minimale : 80%.

Commande de test :

#### ⚙️ Installation & utilisation
##### 1️⃣ Prérequis

Liste les dépendances nécessaires :

Docker (en cours).

##### 2️⃣ Installation

1 git clone https://github.com/Projet-Naturun-the-GOAT/Naturun_The_Goat_Back.git

2 cd Naturun_The_Goat_Back

##### 3️⃣ Lancement du projet

python -m src.python.ai_agent.q_learning

##### 4️⃣ Tests

coverage run -m pytest
