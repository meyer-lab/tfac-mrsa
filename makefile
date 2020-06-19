SHELL := /bin/bash

.PHONY: clean test

flist = 1 2 3 4 5 6
flistFull = $(patsubst %, output/figure%.svg, $(flist))

all: pylint.log $(flistFull) output/manuscript.md

venv: venv/bin/activate
download: tfac/data/mrsa/MRSA.Methylation.txt.xz

venv/bin/activate: requirements.txt tfac/data/mrsa/MRSA.Methylation.txt.xz
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: genFigures.py tfac/figures/figure%.py venv
	@ mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md venv/bin/activate
	. venv/bin/activate && manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	mkdir output/output
	cp output/*.svg output/output/
	. venv/bin/activate && pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml output/manuscript.md

output/manuscript.docx: venv output/manuscript.md $(flistFull)
	. venv/bin/activate && pandoc --verbose -t docx $(pandocCommon) \
		--reference-doc=common/templates/manubot/default.docx \
		--resource-path=.:content \
		-o $@ output/manuscript.md

tfac/data/mrsa/MRSA.Methylation.txt.xz:
	wget -nv -P ./tfac/data/mrsa/ "https://syno.seas.ucla.edu:9001/gc-cytokines/MRSA.Methylation.txt.xz"

test: venv
	. venv/bin/activate && pytest -s

coverage.xml: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov=tfac --cov-report xml:coverage.xml

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc tfac > pylint.log || echo "pylint exited with $?")

clean:
	rm -rf coverage.xml junit.xml output venv
