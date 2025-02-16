install:
	python -m pip install --upgrade pip &&\
		python -m pip install -r requirements.txt

update-requirements:
	python -m pip freeze > requirements.txt

format:
	black ./ && ruff check ./

test_model:
	python -m src.models.test_model

clear-logs: 
	rm logs/*
