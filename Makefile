##
## EPITECH PROJECT, 2026
## my_torch
## File description:
## Builds the C++ my_torch library, copies the Python entry point to the binary name the subject expects
## and puts my_torch_generator next to it
##

ANALYZER	=	my_torch_analyzer
GENERATOR	=	my_torch_generator
MODULE	=	my_torch

all:	module $(ANALYZER) $(GENERATOR)

module:
	$(MAKE) -C $(MODULE)

$(ANALYZER):	$(ANALYZER).py
	cp $< $@
	chmod +x $@

$(GENERATOR):	module
	cp $(MODULE)/$(GENERATOR) $@

clean:
	$(MAKE) -C $(MODULE) clean
	find . -path ./venv -prune -o -type d -name __pycache__ -prune -exec $(RM) -r {} +
	$(RM) -r .pytest_cache .coverage reports

fclean:	clean
	$(MAKE) -C $(MODULE) fclean
	$(RM) $(ANALYZER) $(GENERATOR)

re:	fclean all

tests_run:	module
	$(MAKE) -C $(MODULE) tests_run
	python3 -m pytest

coverage:
	python3 -m pytest --cov --cov-branch --cov-report=term-missing \
		--cov-report=html:reports/htmlcov

.PHONY:	all module clean fclean re tests_run coverage
