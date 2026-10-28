##
## EPITECH PROJECT, 2026
## my_torch
## File description:
## Builds the C++ network as a static library and runs its Criterion tests
##

CXX	=	g++
CXXFLAGS	=	-std=c++20 -Wall -Wextra -Werror -iquote include

SRC	=	src/math/matrix.cpp	\
		src/layers/neuron.cpp	\
		src/activations/activation.cpp	\
		src/layers/layer.cpp	\
		src/layers/dense_layer.cpp	\
		src/network/network.cpp	\
		src/losses/loss.cpp

TESTS	=	tests/math/test_matrix.cpp	\
		tests/layers/test_neuron.cpp	\
		tests/activations/test_activation.cpp	\
		tests/layers/test_layer.cpp	\
		tests/layers/test_dense_layer.cpp	\
		tests/network/test_network.cpp	\
		tests/network/test_gradient_check.cpp	\
		tests/losses/test_loss.cpp

OBJ	=	$(SRC:.cpp=.o)

NAME	=	libmytorch.a
UNIT	=	unit_tests

all:	$(NAME)

$(NAME):	$(OBJ)
	ar rc $@ $^

clean:
	$(RM) $(OBJ) $(shell find . -name '*.gcno' -o -name '*.gcda')

fclean:	clean
	$(RM) $(NAME) $(UNIT)

re:	fclean all

$(UNIT):	$(SRC) $(TESTS)
	$(CXX) $(CXXFLAGS) -iquote tests --coverage -o $@ $^ -lcriterion

tests_run:	$(UNIT)
	./$(UNIT)

.PHONY:	all clean fclean re tests_run
