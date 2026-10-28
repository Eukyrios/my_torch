##
## EPITECH PROJECT, 2026
## my_torch
## File description:
## Builds the C++ network as a static library and runs its Criterion tests
##

CXX	=	g++
CXXFLAGS	=	-std=c++20 -Wall -Wextra -Werror -iquote include

SRC	=	src/neuron.cpp	\
		src/matrix.cpp	\
		src/dense.cpp

TESTS	=	tests/test_neuron.cpp	\
		tests/test_matrix.cpp	\
		tests/test_dense.cpp

OBJ	=	$(SRC:.cpp=.o)

NAME	=	libmytorch.a
UNIT	=	unit_tests

all:	$(NAME)

$(NAME):	$(OBJ)
	ar rc $@ $^

clean:
	$(RM) $(OBJ) *.gcno *.gcda

fclean:	clean
	$(RM) $(NAME) $(UNIT)

re:	fclean all

$(UNIT):	$(SRC) $(TESTS)
	$(CXX) $(CXXFLAGS) --coverage -o $@ $^ -lcriterion

tests_run:	$(UNIT)
	./$(UNIT)

.PHONY:	all clean fclean re tests_run
