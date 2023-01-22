#include "../include/utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#define N 10000000
#define K 4

float *pontoX __attribute__((aligned(4))), *pontoY __attribute__((aligned(4)));
float *centroidX __attribute__((aligned(4))), *centroidY __attribute__((aligned(4)));
int *pontoCluster __attribute__((aligned(4)));
bool terminado = false;
int iteracoes = 0;

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
    //Para cada ponto
    for(int i = 0; i < N; i++) {

        float minDistancia = FP_INFINITE;
        int bestCluster = -1;
        //Calcular a distância entre todos os centroides.
        for(int j = 0; j < K; j++) {
            //Distância euclidiana
            float xSub = (pontoX[i] - centroidX[j]);
            float ySub = (pontoY[i] - centroidY[j]);
            float distancia = xSub*xSub + ySub*ySub;
            if (distancia < minDistancia){
                minDistancia = distancia;
                bestCluster = j;
            }
        }

        //Se houve mudança de cluster, marca como instável.
        if(pontoCluster[i] != bestCluster)
            terminado = false;

        //Atribuir o melhor cluster ao ponto
        pontoCluster[i] = bestCluster;
    }
}

void ajustaCentroides(){

    //Arrays com somatórios e número de pontos
    float sumValX[K], sumValY[K];
    int numPontos[K];

    for(int i = 0; i < K; i++) {
        numPontos[i] = 0;
        sumValY[i] = 0;
        sumValX[i] = 0;
    }

    //Para cada ponto, adiciona as coordenadas no respetivo somatório
    for(int i = 0; i < N; i++) {
        int idCluster = pontoCluster[i];

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
    //Cálculo o número de pontos por clusters
    int numPontosInCluster[K];
    for(int i = 0; i < K; i++)
        numPontosInCluster[i] = 0;

    for(int i = 0; i < N; i++)
        numPontosInCluster[pontoCluster[i]]++;

    //Imprime toda a informação
    printf("N = %d, K = %d\n",N,K);
    for(int i = 0; i < K; i++)
        printf("Cluster %d: (%f,%f). Size:%d\n",i,centroidX[i],centroidY[i],numPontosInCluster[i]);

    printf("NºIterações: %d\n",iteracoes);
}

int main(){
    aloca();
    inicializa();

    atribuiClusters();
    while(!terminado){
        terminado = true;
        ajustaCentroides();
        atribuiClusters();
        iteracoes++;
    }

    printFinalResult();
}
