SHELL := /bin/bash

.PHONY: clean test

flist = 1 2 3 4 5 6
flistFull = $(patsubst %, output/figure%.svg, $(flist))
pandocCommon = -f markdown \
	--bibliography=output/references.json \
	--csl=style.csl -F pandoc-fignos -F pandoc-eqnos -F pandoc-tablenos \
	--metadata link-citations=true

all: pylint.log $(flistFull) output/manuscript.html output/manuscript.pdf output/manuscript.docx

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: venv genFigures.py tfac/figures/figure%.py
	mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

output/manuscript.md: venv manuscript/*.md
	mkdir -p ./output/%
	. venv/bin/activate && manubot process --content-directory=manuscript/ --output-directory=output/ --log-level=WARNING

output/manuscript.html: venv output/manuscript.md style.csl $(flistFull)
	. venv/bin/activate && pandoc \
		--from=markdown --to=html5 --filter=pandoc-fignos --filter=pandoc-eqnos --filter=pandoc-tablenos \
		--bibliography=output/$*/references.json \
		--csl=style.csl \
		--metadata link-citations=true \
		--include-after-body=common/templates/manubot/default.html \
		--include-after-body=common/templates/manubot/plugins/table-scroll.html \
		--include-after-body=common/templates/manubot/plugins/anchors.html \
		--include-after-body=common/templates/manubot/plugins/accordion.html \
		--include-after-body=common/templates/manubot/plugins/tooltips.html \
		--include-after-body=common/templates/manubot/plugins/jump-to-first.html \
		--include-after-body=common/templates/manubot/plugins/link-highlight.html \
		--include-after-body=common/templates/manubot/plugins/table-of-contents.html \
		--include-after-body=common/templates/manubot/plugins/lightbox.html \
		--mathjax \
		--variable math="" \
		--include-after-body=common/templates/manubot/plugins/math.html \
		--include-after-body=common/templates/manubot/plugins/hypothesis.html \
		--output=output/manuscript.html output/manuscript.md

output/manuscript.pdf: venv output/manuscript.md $(flistFull) style.csl
	. venv/bin/activate && pandoc -t html5 $(pandocCommon) \
	--pdf-engine=weasyprint --pdf-engine-opt=--presentational-hints \
	--webtex=https://latex.codecogs.com/svg.latex? \
	--include-after-body=common/templates/manubot/default.html \
	-o $@ output/manuscript.md

output/manuscript.docx: venv output/manuscript.md $(flistFull) style.csl
	. venv/bin/activate && pandoc --verbose -t docx $(pandocCommon) \
		--reference-doc=common/templates/manubot/default.docx \
		--resource-path=.:content \
		-o $@ output/manuscript.md

test: venv
	. venv/bin/activate; pytest -s

coverage.xml: venv
	. venv/bin/activate; pytest --junitxml=junit.xml --cov=tfac --cov-report xml:coverage.xml

pylint.log: venv
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc tfac > pylint.log || echo "pylint exited with $?")

style.csl:
	curl -o $@ https://www.zotero.org/styles/plos-computational-biology

clean:
	mv output/requests-cache.sqlite requests-cache.sqlite || true
	rm -rf coverage.xml junit.xml output venv style.csl
	mkdir output
	mv requests-cache.sqlite output/requests-cache.sqlite || true
