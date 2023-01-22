# CP - Computação Paralela 2022/2023 - TP3 - Algoritmo K-means With CUDA

## Trabalho Realizado por:

- Vicente Gonçalves Moreira - PG50799
- Joana Maia Teixeira Alves - PG50457

## Instruções de utilização:

### Compilação

#### Requisitos

- GNU Make (>= V3.82)
- GCC (>= V7.2.0)
- Nvidia CUDA compilier (>= V10.1.243)

Na pasta fonte, correr o comando:

```
 make
```

### Execução

- Execução Simples:

```
 make run
```

- Execução PerfStat:

```
 make runPerfStat
```

- Execução Nvidia Profiler:

```
 make runNvprof
```

Por defeito serão utilizados os valores N = 1.000.000 e K = 1024. Para alterar estes valores, na execução dos comandos make, é possível modificar estes valores ao adicionar flags tais como:

```
 make run N=10000 K=32
 make run K=512
```

### Execução no Cluster Search

Estão dispobinilizados dois scripts (ClusterRun.sh e ClusterProfiling.sh), que executam os comandos acima ('make run' e 'make runNvprof', respetivamente), utilizando as váriáveis de ambiente necessárias do Search, sendo só necessário executar:

```
 sbatch ./ClusterRun.sh
```

(Podem ser fornecidos argumentos de N e K como indicado acima)
