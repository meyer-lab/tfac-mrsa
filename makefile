SHELL := /bin/bash

.PHONY: clean test

flist = $(wildcard tfac/figures/figure*.py)

all: $(patsubst tfac/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: tfac/figures/figure%.py
	@ mkdir -p ./output
	poetry run fbuild $*

test:
	poetry run pytest -s -x -v --full-trace

clean:
	rm -rf coverage.xml junit.xml
	git clean -ffdx output
