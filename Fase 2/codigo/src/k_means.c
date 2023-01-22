#include "../include/utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

float *pontoX __attribute__((aligned(4))), *pontoY __attribute__((aligned(4)));
float *centroidX __attribute__((aligned(4))), *centroidY __attribute__((aligned(4)));
int *pontoCluster __attribute__((aligned(4)));

int N,K,THREADS;
int nIter = 0;
int nMaxIter = 20;

void aloca(){
    pontoX = (float *) malloc(N*sizeof(float));
    pontoY = (float *) malloc(N*sizeof(float));
    centroidX = (float *) malloc(K*sizeof(float));
    centroidY = (float *) malloc(K*sizeof(float));
    pontoCluster = (int *) malloc(N*sizeof(int));
}

void inicializa() {
    srand(10);
    for(int i = 0; i < N; i++) {
        pontoX[i] = (float) rand() / RAND_MAX;
        pontoY[i] = (float) rand() / RAND_MAX;
        pontoCluster[i] = -1;
    }
    for(int i = 0; i < K; i++) {
        centroidX[i] = pontoX[i];
        centroidY[i] = pontoY[i];
    }
}


void atribuiClusters(){
    float minDistancia,xSub,ySub,distancia;
    int bestCluster;

    //Para cada ponto
    #pragma omp parallel for num_threads(THREADS) schedule(static) private(minDistancia,bestCluster,xSub,ySub,distancia)
    for(int i = 0; i < N; i++) {

        minDistancia = FP_INFINITE;
        bestCluster = -1;
        //Calcular a distância entre todos os centroides.
        for(int j = 0; j < K; j++) {
            //Distância euclidiana
            xSub = (pontoX[i] - centroidX[j]);
            ySub = (pontoY[i] - centroidY[j]);
            distancia = xSub*xSub + ySub*ySub;
            if (distancia < minDistancia){
                minDistancia = distancia;
                bestCluster = j;
            }
        }

        //Atribuir o melhor cluster ao ponto
        pontoCluster[i] = bestCluster;
    }

}

void ajustaCentroides(){

    //Arrays com somatórios e número de pontos
    float sumValX[K] __attribute__((aligned(4)));
    float sumValY[K] __attribute__((aligned(4)));
    int numPontos[K] __attribute__((aligned(4)));

    for(int i = 0; i < K; i++) {
        numPontos[i] = 0;
        sumValY[i] = 0;
        sumValX[i] = 0;
    }

    //Para cada ponto, adiciona as coordenadas no respetivo somatório
    int idCluster;
    #pragma omp parallel for num_threads(THREADS) schedule(static) private(idCluster) reduction(+:sumValX) reduction(+:sumValY) reduction(+:numPontos)
    for(int i = 0; i < N; i++) {
        idCluster = pontoCluster[i];

        sumValX[idCluster] += pontoX[i];
        sumValY[idCluster] += pontoY[i];
        numPontos[idCluster]++;
    }

    //Cálcula as novas médias
    for(int i = 0; i < K; i++) {
        centroidX[i] = sumValX[i] / numPontos[i];
        centroidY[i] = sumValY[i] / numPontos[i];
    }

}

void printFinalResult(){
    //Cálculo do número de pontos por clusters
    int numPontosInCluster[K];
    for(int i = 0; i < K; i++)
        numPontosInCluster[i] = 0;

    for(int i = 0; i < N; i++)
        numPontosInCluster[pontoCluster[i]]++;

    //Imprime toda a informação
    printf("N = %d, K = %d\n",N,K);
    for(int i = 0; i < K; i++)
        printf("Center: (%.3f, %.3f) : Size: %d\n",centroidX[i],centroidY[i],numPontosInCluster[i]);

    printf("Iterations: %d\n",nIter);
}

int main(int argc,char** argv){
    if (argc != 4 && argc != 3){
        printf("Número de argumentos incorretos\n");
        return 1;
    }

    int argNums[3] = {0,0,1};
    for(int i = 1;i<argc;i++){
        if (sscanf(argv[i],"%d",&argNums[i-1]) != 1){
            printf("Invalid argument '%s'. Only Numbers accepted\n",argv[i]);
            return 1;
        }
    }
    N = argNums[0];
    K = argNums[1];
    THREADS = argNums[2];

    aloca();
    inicializa();

    atribuiClusters();
    while(nIter < nMaxIter){
        ajustaCentroides();
        atribuiClusters();
        nIter++;
    }

    printFinalResult();
}
