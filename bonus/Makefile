##
## EPITECH PROJECT, 2026
## my_torch
## File description:
## Bonus: rebuilds the stalemate dataset and trains the 6-class network from it, with the
## binaries of the project (run make at the root first)
##

ROOT	=	..
DATA	=	$(ROOT)/data/stalemate
NETWORK	=	my_torch_network_stalemate.nn
CONFIG	=	stalemate_network.conf
PYTHON	=	python3

all:	$(NETWORK)

dataset:	$(DATA)/train.txt

$(DATA)/train.txt:	stalemate_dataset.py
	cd $(ROOT) && $(PYTHON) -m bonus.stalemate_dataset --data data --out data/stalemate --per-class 40000 --seed 42

$(NETWORK):	$(CONFIG) $(DATA)/train.txt
	$(ROOT)/my_torch_generator $(CONFIG) 1
	$(ROOT)/my_torch_analyzer --train --save $@ stalemate_network_1.nn $(DATA)/train.txt
	$(RM) stalemate_network_1.nn

evaluate:	$(NETWORK)
	$(ROOT)/my_torch_analyzer --predict $(NETWORK) $(DATA)/test.txt > /dev/null

clean:
	$(RM) stalemate_network_1.nn

fclean:	clean
	$(RM) -r $(DATA)

re:	fclean all

.PHONY:	all dataset evaluate clean fclean re
