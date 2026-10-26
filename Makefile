##
## EPITECH PROJECT, 2026
## my_torch
## File description:
## Copies the Python entry point to the binary name the subject expects
##

ANALYZER	=	my_torch_analyzer

all:	$(ANALYZER)

$(ANALYZER):	$(ANALYZER).py
	cp $< $@
	chmod +x $@

clean:
	find . -path ./venv -prune -o -type d -name __pycache__ -prune -exec $(RM) -r {} +
	$(RM) -r .pytest_cache .coverage reports

fclean:	clean
	$(RM) $(ANALYZER)

re:	fclean all

tests_run:
	python3 -m pytest

coverage:
	python3 -m pytest --cov --cov-branch --cov-report=term-missing \
		--cov-report=html:reports/htmlcov

.PHONY:	all clean fclean re tests_run coverage
