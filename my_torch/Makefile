##
## EPITECH PROJECT, 2026
## my_torch
## File description:
## Builds the C++ network as a static library, runs its Criterion tests and builds the benchmarks
##

CXX	=	g++
CXXFLAGS	=	-std=c++20 -Wall -Wextra -Werror -iquote src

SRC	=	src/math/matrix.cpp	\
		src/model/neuron/neuron.cpp	\
		src/model/initialization/initialization.cpp	\
		src/model/activations/activation.cpp	\
		src/model/activations/activations/linear/linear.cpp	\
		src/model/activations/activations/relu/relu.cpp	\
		src/model/activations/activations/sigmoid/sigmoid.cpp	\
		src/model/activations/activations/tanh/tanh.cpp	\
		src/model/activations/activations/softmax/softmax.cpp	\
		src/model/layers/layer.cpp	\
		src/model/layers/layers/dense_layer/dense_layer.cpp	\
		src/model/layers/layers/dropout/dropout.cpp	\
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
		src/training/optimizers/optimizers/momentum/momentum.cpp	\
		src/training/training.cpp	\
		src/io/json/json.cpp	\
		src/io/npy/npy.cpp	\
		src/io/zip/zip.cpp	\
		src/io/network_file/network_file.cpp	\
		src/io/boards/boards.cpp	\
		src/engine/engine.cpp	\
		src/training/hyperparameters/search_space/search_space.cpp	\
		src/training/hyperparameters/algorithms/particle_swarm/particle_swarm.cpp	\
		src/training/hyperparameters/algorithms/genetic_algorithm/genetic_algorithm.cpp	\
		src/training/hyperparameters/validation_loss/validation_loss.cpp	\
		src/training/hyperparameters/search_log/search_log.cpp

TESTS	=	tests/math/test_matrix.cpp	\
		tests/model/neuron/test_neuron.cpp	\
		tests/model/initialization/test_initialization.cpp	\
		tests/model/activations/test_activation.cpp	\
		tests/model/layers/test_layer.cpp	\
		tests/model/layers/test_dense_layer.cpp	\
		tests/model/layers/test_dropout.cpp	\
		tests/model/network/test_network.cpp	\
		tests/model/network/test_gradient_check.cpp	\
		tests/model/network/test_training.cpp	\
		tests/model/network/test_xor.cpp	\
		tests/training/losses/test_loss.cpp	\
		tests/training/schedules/test_schedule.cpp	\
		tests/training/optimizers/test_optimizer.cpp	\
		tests/training/test_training_runs.cpp	\
		tests/training/test_early_stopping.cpp	\
		tests/io/json/test_json.cpp	\
		tests/io/npy/test_npy.cpp	\
		tests/io/zip/test_zip.cpp	\
		tests/io/network_file/test_network_file.cpp	\
		tests/io/boards/test_boards.cpp	\
		tests/engine/test_engine.cpp	\
		tests/training/hyperparameters/test_search_space.cpp	\
		tests/training/hyperparameters/test_particle_swarm.cpp	\
		tests/training/hyperparameters/test_genetic_algorithm.cpp	\
		tests/training/hyperparameters/test_validation_loss.cpp	\
		tests/training/hyperparameters/test_search_log.cpp

OBJ	=	$(SRC:.cpp=.o)

NAME	=	libmytorch.a
UNIT	=	unit_tests
BENCHMARK	=	hyperparameter_benchmark
ENGINE	=	my_torch_engine

all:	$(NAME) $(ENGINE)

$(NAME):	$(OBJ)
	ar rc $@ $^

clean:
	$(RM) $(OBJ) $(shell find . -name '*.gcno' -o -name '*.gcda')

fclean:	clean
	$(RM) $(NAME) $(UNIT) $(BENCHMARK) $(ENGINE)

re:	fclean all

$(UNIT):	$(SRC) $(TESTS)
	$(CXX) $(CXXFLAGS) -iquote tests --coverage -o $@ $^ -lcriterion

tests_run:	$(UNIT)
	./$(UNIT)

$(BENCHMARK):	$(SRC) benchmarks/hyperparameter_search.cpp
	$(CXX) $(CXXFLAGS) -O2 -o $@ $^

benchmark:	$(BENCHMARK)

# run by the Python analyzer; optimised, as it trains the networks
$(ENGINE):	$(SRC) engine/main.cpp
	$(CXX) $(CXXFLAGS) -O2 -o $@ $^

.PHONY:	all clean fclean re tests_run benchmark
