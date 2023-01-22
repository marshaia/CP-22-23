################################################################################
# Makefile for general code snippets
#
# by André Pereira
# extended by Vicente Moreira & Joana Alves
################################################################################

SHELL = /bin/sh
BIN_NAME = k_means

CXX = nvcc
LD  = nvcc

CXXFLAGS   = -O2 -g -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets 

SRC_DIR = src
BIN_DIR = bin
BUILD_DIR = build
SRC = $(wildcard $(SRC_DIR)/*.cu)
OBJ = $(patsubst src/%.cu,build/%.o,$(SRC))
BIN = $(BIN_NAME)

vpath %.cu $(SRC_DIR)

################################################################################
# Rules
################################################################################

.DEFAULT_GOAL = all

$(BUILD_DIR)/%.o: %.cu
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

$(BIN_DIR)/$(BIN_NAME): $(OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJ) $(LIBS)

checkdirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

all: checkdirs $(BIN_DIR)/$(BIN_NAME)

clean:
	rm -f $(BUILD_DIR)/* $(BIN_DIR)/* 


#Default values for k_means algorithm
N = 1000000
K = 1024

run: checkdirs $(BIN_DIR)/$(BIN_NAME)
	./$(BIN_DIR)/$(BIN_NAME) $(N) $(K)

runPerfStat: checkdirs $(BIN_DIR)/$(BIN_NAME)
	perf stat -o perfStat.txt -e L1-dcache-load-misses -M cpi ./$(BIN_DIR)/$(BIN_NAME) $(N) $(K) > output.txt

runNvprof: checkdirs $(BIN_DIR)/$(BIN_NAME)
	nvprof ./$(BIN_DIR)/$(BIN_NAME) $(N) $(K) > output.txt
