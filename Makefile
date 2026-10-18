install:
	pip install -r requirements-dev.txt

build:
	python -m prahari build

test:
	python -m pytest

eval:
	python -m eval.run

run:
	uvicorn prahari.api:get_app --factory --reload --port 8000

up:
	docker compose up --build

.PHONY: install build test eval run up
