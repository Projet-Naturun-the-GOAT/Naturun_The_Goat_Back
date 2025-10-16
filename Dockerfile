# Étape 1 : image de base
FROM python:3.10

# Étape 2 : définir le répertoire de travail
WORKDIR /src

# Étape 3 : copier les fichiers
COPY . /src

# Étape 4 : installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : commande à exécuter
CMD ["python", "-m", "src.python.ai_agent.q_learning"]