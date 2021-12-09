SHELL := /bin/bash

.PHONY: clean test

flist = 2 3 4 5 6 7 S1
flistFull = $(patsubst %, output/figure%.svg, $(flist))

all: $(flistFull)

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: genFigures.py tfac/figures/figure%.py venv
	@ mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

test: venv
	. venv/bin/activate && pytest -s -x -v

coverage.xml: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov=tfac --cov-report xml:coverage.xml

clean:
	rm -rf coverage.xml junit.xml venv
	git clean -f output
