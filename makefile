SHELL := /bin/bash

.PHONY: clean test

flist = 2 3 4 5 S1
flistFull = $(patsubst %, output/figure%.svg, $(flist))

all: pylint.log output/manuscript.md $(flistFull)

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: genFigures.py tfac/figures/figure%.py venv
	@ mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md venv/bin/activate
	. venv/bin/activate && manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md $(flistFull)
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml output/manuscript.md

output/manuscript.docx: venv output/manuscript.md $(flistFull)
	. venv/bin/activate && pandoc --verbose -t docx $(pandocCommon) \
		--reference-doc=common/templates/manubot/default.docx \
		--resource-path=.:content \
		-o $@ output/manuscript.md

test: venv
	. venv/bin/activate && pytest -s -x -v

coverage.xml: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov=tfac --cov-report xml:coverage.xml

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc tfac > pylint.log || echo "pylint exited with $?")

clean:
	rm -rf coverage.xml junit.xml venv
	git clean -f output
