# On prend EXACTEMENT ta version
FROM python:3.12.4-slim

WORKDIR /app

# Install dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY . .

# TA COMMANDE (Note le 0.0.0.0 obligatoire pour Docker)
CMD ["python", "-m", "uvicorn", "api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]