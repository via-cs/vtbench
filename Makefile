.DEFAULT_GOAL := help

# CLEAN

.PHONY: clean
clean:
	-rm -rf build dist .pytest_cache .tox *.egg-info __pycache__ vtbench/__pycache__ tests/__pycache__

.PHONY: install 
install: clean
	pip install .

.PHONY: install-dev 
install-dev: clean
	pip install -e .[dev]

# TESTING

.PHONY: test
test: 
	pytest


# LINTING

.PHONY: lint
lint: 
	flake8 vtbench tests

# VERSION BUMPING

.PHONY: bump
bump: 
	bump2version patch

# DOCS

.PHONY: docs
docs: 
	sphinx-build -b html docs/ docs/_build/html

.PHONY: view-docs
view-docs: docs
	 python -c "import webbrowser, os; webbrowser.open('file://' + os.path.abspath('docs/_build/html/index.html'))"


# RELEASE
.PHONY: dist
dist: clean
	python setup.py sdist
	python setup.py bdist_wheel

.PHONY: publish-test
publish-test: dist
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: publish 
publish: dist
	twine upload dist/*
