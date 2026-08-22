.PHONY: all analyses figures test paper clean

all: analyses figures test

analyses:
	python scripts/analyse_gap.py
	python scripts/analyse_recoverability.py
	python scripts/analyse_cost.py
	python scripts/analyse_carp_geometry.py

figures:
	python scripts/make_figures.py

test:
	pytest -q

paper: figures
	cd paper && latexmk -pdf -interaction=nonstopmode paper.tex

clean:
	cd paper && latexmk -C
	rm -rf .pytest_cache **/__pycache__
