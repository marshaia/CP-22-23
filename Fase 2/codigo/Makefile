CC = gcc
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means
THREADS = 10

CFLAGS = -O2 -Wall -fopenmp

.DEFAULT_GOAL = k_means

k_means: $(SRC)k_means.c $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)k_means.c $(BIN)utils.o -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*

runseq:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)

runpar:
	./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)

runPerfStat:
	perf stat -o perfStat.txt -e L1-dcache-load-misses -M cpi ./$(BIN)$(EXEC) 10000000 $(K) $(T) > output.txt

runInScript:
	perf stat -o ./logFold/StatNThreads$(T)Test$(NUM).txt -e L1-dcache-load-misses -M cpi ./$(BIN)$(EXEC) 10000000 32 $(T) > ./logFold/outputT$(T).txt
