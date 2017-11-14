OBJS_SHARED = vmf_sampler.o logit_sampler.o gaussian_sampler.o sampler.o parse_args.o utils.o
OBJS_MPI = $(OBJS_SHARED) barycenter_mpi.o
CC = mpic++
CFLAGS = -Wall -Wextra -Wformat=2 -c -g -O0 -std=c++11 -march=native -pthread
LFLAGS = -Wall -Wextra -Wformat=2 -g -O0 -std=c++11 -march=native -pthread
LIB = -lhdf5_cpp -lhdf5
INCLUDE = -I/usr/include/eigen3/ -Icxxopts/include/ -Ieigen3-hdf5/

barycenter_mpi: $(OBJS_MPI)
	$(CC) $(LFLAGS) -o barycenter_mpi $(OBJS_MPI) $(INCLUDE) $(LIB)

sampler.o: sampler.cpp sampler.hpp
	$(CC) $(CFLAGS) sampler.cpp $(INCLUDE) $(LIB)

vmf_sampler.o: vmf_sampler.cpp vmf_sampler.hpp sampler.hpp
	$(CC) $(CFLAGS) vmf_sampler.cpp $(INCLUDE) $(LIB)

logit_sampler.o: logit_sampler.cpp logit_sampler.hpp sampler.hpp
	$(CC) $(CFLAGS) logit_sampler.cpp $(INCLUDE) $(LIB)

gaussian_sampler.o: gaussian_sampler.cpp gaussian_sampler.hpp sampler.hpp
	$(CC) $(CFLAGS) gaussian_sampler.cpp $(INCLUDE) $(LIB)

parse_args.o: parse_args.cpp parse_args.hpp
	$(CC) $(CFLAGS) parse_args.cpp $(INCLUDE) $(LIB)

utils.o: utils.cpp utils.hpp
	$(CC) $(CFLAGS) utils.cpp $(INCLUDE) $(LIB)

barycenter_mpi.o: barycenter_mpi.cpp sampler.hpp vmf_sampler.hpp logit_sampler.hpp gaussian_sampler.hpp parse_args.hpp utils.hpp
	$(CC) $(CFLAGS) barycenter_mpi.cpp $(INCLUDE) $(LIB)

clean:
	rm -f *.o \
	barycenter_mpi
