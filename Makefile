#################################################################################
# Optimisation tarifaire -- marketplace Olist                                   #
#################################################################################

PROJECT_NAME = optimization_pricing
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# ENVIRONNEMENT                                                                 #
#################################################################################

## Installer les dependances
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Creer l'environnement virtuel
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo ">>> Environnement cree. Activation :"
	@echo ">>>   Windows : .venv\Scripts\activate"
	@echo ">>>   Unix    : source .venv/bin/activate"

#################################################################################
# PIPELINE DE DONNEES ET DE MODELISATION                                        #
#################################################################################

## 1. Construire les panels (categorie x semaine et produit x mois)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m optimization_pricing.dataset

## 2. Construire les features
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m optimization_pricing.features

## 3. Estimer les elasticites-prix (tous les estimateurs)
.PHONY: elasticity
elasticity:
	$(PYTHON_INTERPRETER) -m optimization_pricing.elasticity

## 4. Entrainer les modeles de demande et generer la grille tarifaire
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m optimization_pricing.modeling.train

## 5. Generer le rapport decisionnel et les figures
.PHONY: report
report:
	$(PYTHON_INTERPRETER) -m optimization_pricing.reporting

## Pipeline complet, des CSV bruts au rapport
.PHONY: pipeline
pipeline: data features elasticity train report

#################################################################################
# SERVICES                                                                      #
#################################################################################

## Lancer l'API en local (http://localhost:8000/docs)
.PHONY: api
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

## Lancer le dashboard en local (http://localhost:8501)
.PHONY: dashboard
dashboard:
	streamlit run streamlit_app/app.py

## Ouvrir l'interface MLflow (http://localhost:5000)
.PHONY: mlflow
mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

## Construire et lancer API + dashboard via Docker
.PHONY: docker-up
docker-up:
	docker compose up --build -d
	@echo ">>> API       : http://localhost:8000/docs"
	@echo ">>> Dashboard : http://localhost:8501"

## Arreter les conteneurs
.PHONY: docker-down
docker-down:
	docker compose down

#################################################################################
# QUALITE                                                                       #
#################################################################################

## Lancer les tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests -v

## Verifier le style (sans modifier)
.PHONY: lint
lint:
	ruff check .
	ruff format --check .

## Corriger le style
.PHONY: format
format:
	ruff check --fix .
	ruff format .

## Supprimer les fichiers compiles
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# AIDE                                                                          #
#################################################################################

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Commandes disponibles :\n'); \
print('\n'.join(['{:22}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

.DEFAULT_GOAL := help

help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
