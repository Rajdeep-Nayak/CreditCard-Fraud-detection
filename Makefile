.PHONY: install data data-synthetic train train-all promote registry-show export-champion serve ui demo-samples test lint mlflow-ui

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

promote:
	python -m fraudguard.registry promote

registry-show:
	python -m fraudguard.registry show

export-champion:
	python -m fraudguard.registry export --out artifacts/champion

serve:
	uvicorn fraudguard.api.main:app --reload

ui:
	streamlit run app/streamlit_app.py

demo-samples:
	python scripts/make_demo_samples.py

test:
	pytest

lint:
	ruff check src tests scripts

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
