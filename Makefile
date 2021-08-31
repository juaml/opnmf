# Makefile before PR
#

.PHONY: checks

checks: flake spellcheck

flake:
	flake8

spellcheck:
	codespell opnmf/ docs/ examples/

test:
	pytest -v