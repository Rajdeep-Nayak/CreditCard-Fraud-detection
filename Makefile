.PHONY: install data data-synthetic train train-all test lint mlflow-ui

install:
	pip install -r requirements-dev.txt && pip install -e .

data:            ## needs Kaggle credentials (~/.kaggle/kaggle.json)
	python scripts/download_data.py

data-synthetic:  ## quick demo data, NOT for reporting results
	python scripts/download_data.py --synthetic

train:
	python -m fraudguard.train --config configs/train.yaml

train-all:
	python -m fraudguard.train --config configs/train.yaml --models logreg,rf,hgb

test:
	pytest

lint:
	ruff check src tests scripts

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
