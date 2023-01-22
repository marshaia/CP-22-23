CC       = gcc
BIN      = bin/
SRC      = src/
INCLUDES = include/
EXEC     = k_means

CFLAGS = -O2 -Wall -ftree-vectorize -msse4

.DEFAULT_GOAL = k_means

k_means: $(SRC)k_means.c $(BIN)utils.o
	$(CC) $(CFLAGS) $(SRC)k_means.c $(BIN)utils.o -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(CC) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o

clean:
	rm -r bin/*

run:
	./$(BIN)$(EXEC)

runPerfStat:
	perf stat -o perfStat.txt -e L1-dcache-load-misses -M cpi ./$(BIN)$(EXEC) > output.txt
