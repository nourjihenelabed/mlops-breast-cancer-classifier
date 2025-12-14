PYTHON=python
ENV=mlops_env

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) main.py

mlflow:
	mlflow ui

api:
	uvicorn api.app:app --reload

docker-build:
	docker build -t breast-cancer-api .

docker-run:
	docker run -p 8000:8000 breast-cancer-api

clean:
	rm -rf __pycache__ .pytest_cache mlruns
