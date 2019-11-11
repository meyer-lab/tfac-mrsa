SHELL := /bin/bash

.PHONY: clean test

flist = 1 2 3 4 5 6

all: pylint.log $(patsubst %, output/figure%.svg, $(flist))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Ur requirements.txt
	touch venv/bin/activate

output/figure%.svg: venv genFigures.py tfac/figures/figure%.py
	mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

test: venv
	. venv/bin/activate; pytest -s

coverage.xml: venv
	. venv/bin/activate; pytest --junitxml=junit.xml --cov=tfac --cov-report xml:coverage.xml

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc tfac > pylint.log || echo "pylint exited with $?")

clean:
	rm -rf coverage.xml junit.xml output venv
