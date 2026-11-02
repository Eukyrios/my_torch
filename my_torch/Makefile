##
## EPITECH PROJECT, 2026
## my_torch
## File description:
## Builds the C++ network as a static library and runs its Criterion tests
##

CXX	=	g++
CXXFLAGS	=	-std=c++20 -Wall -Wextra -Werror -iquote src

SRC	=	src/math/matrix.cpp	\
		src/model/neuron/neuron.cpp	\
		src/model/activations/activation.cpp	\
		src/model/activations/activations/linear/linear.cpp	\
		src/model/activations/activations/relu/relu.cpp	\
		src/model/activations/activations/sigmoid/sigmoid.cpp	\
		src/model/activations/activations/tanh/tanh.cpp	\
		src/model/activations/activations/softmax/softmax.cpp	\
		src/model/layers/layer.cpp	\
		src/model/layers/layers/dense_layer/dense_layer.cpp	\
		src/model/network/network.cpp	\
		src/training/losses/loss.cpp	\
		src/training/losses/losses/mean_squared_error/mean_squared_error.cpp	\
		src/training/losses/losses/cross_entropy/cross_entropy.cpp	\
		src/training/schedules/schedule.cpp	\
		src/training/schedules/schedules/constant/constant.cpp	\
		src/training/schedules/schedules/time_decay/time_decay.cpp	\
		src/training/schedules/schedules/exponential_decay/exponential_decay.cpp	\
		src/training/schedules/schedules/step_decay/step_decay.cpp	\
		src/training/optimizers/optimizer.cpp	\
		src/training/optimizers/optimizers/sgd/sgd.cpp	\
		src/training/optimizers/optimizers/momentum/momentum.cpp

TESTS	=	tests/math/test_matrix.cpp	\
		tests/model/neuron/test_neuron.cpp	\
		tests/model/activations/test_activation.cpp	\
		tests/model/layers/test_layer.cpp	\
		tests/model/layers/test_dense_layer.cpp	\
		tests/model/network/test_network.cpp	\
		tests/model/network/test_gradient_check.cpp	\
		tests/model/network/test_training.cpp	\
		tests/model/network/test_xor.cpp	\
		tests/training/losses/test_loss.cpp	\
		tests/training/schedules/test_schedule.cpp	\
		tests/training/optimizers/test_optimizer.cpp

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
