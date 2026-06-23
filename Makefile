.PHONY: setup install install-dev test lint fmt run-api run-worker run-ui eval redteam demo docker

setup:
	bash scripts/setup.sh

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check .

fmt:
	ruff check --fix .

run-api:
	uvicorn app.api:app --reload --port 8000

run-worker:
	arq app.worker.WorkerSettings

run-ui:
	streamlit run app/ui_streamlit.py

eval:
	python -m eval.run_eval

redteam:
	python -m eval.redteam

demo:
	DEMO_MODE=1 streamlit run app/ui_streamlit.py

docker:
	docker compose up --build
