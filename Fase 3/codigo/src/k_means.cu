#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "./k_means.h"

//Host Memory
float *pontoX,*pontoY;
float *centroidX,*centroidY;
int *pontoCluster;

//Device Memory
float *dpontoX, *dpontoY;
float *dcentroidX, *dcentroidY;
int *dpontoCluster;

float *dsumX,*dsumY;
int *dnumPontos;

//Constants and Blocks Size
#define NUM_THREADS_PER_BLOCK 1024
int NUM_BLOCKS_FOR_POINTS,NUM_BLOCKS_FOR_CENTROIDS;
int POINT_ARRAY_SIZE,CENTROID_ARRAY_SIZE,POINTCLUSTER_ARRAY_SIZE;
int N,K;
int nIter = 0;
int nMaxIter = 20;


using namespace std;

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


// -----------------------------------
// -------------  CUDA ---------------
// -----------------------------------
void initializeKernel (){
    // allocate the memory on the device
    cudaMalloc ((void**) &dpontoX, POINT_ARRAY_SIZE);
    cudaMalloc ((void**) &dpontoY, POINT_ARRAY_SIZE);
    cudaMalloc ((void**) &dcentroidX, CENTROID_ARRAY_SIZE);
    cudaMalloc ((void**) &dcentroidY, CENTROID_ARRAY_SIZE);
    cudaMalloc ((void**) &dpontoCluster, POINTCLUSTER_ARRAY_SIZE);
    cudaMalloc ((void**) &dsumX, CENTROID_ARRAY_SIZE);
    cudaMalloc ((void**) &dsumY, CENTROID_ARRAY_SIZE);
    cudaMalloc ((void**) &dnumPontos,  K * sizeof(int));
    checkCUDAError("mem allocation");

    // Copy all working data to Device
    cudaMemcpy (dpontoX,pontoX,POINT_ARRAY_SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy (dpontoY,pontoY,POINT_ARRAY_SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy (dcentroidX,centroidX,CENTROID_ARRAY_SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy (dcentroidY,centroidY,CENTROID_ARRAY_SIZE,cudaMemcpyHostToDevice);
    checkCUDAError("memcpy h->d");
}

void freeKernel (){
    // free the device memory
    cudaFree(dpontoX); cudaFree(dpontoY);
    cudaFree(dcentroidX); cudaFree(dcentroidY);
    cudaFree(dpontoCluster);
    cudaFree(dsumX); cudaFree(dsumY);
    cudaFree(dnumPontos);
    checkCUDAError("mem free");
}

__global__
void atribuiClusterKernel (float *myPontoX, float *myPontoY, int myN, float *myCentroidX, float *myCentroidY, int myK, int *myPontoCluster) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    
    __shared__ float shared_centroidX[NUM_THREADS_PER_BLOCK];
    __shared__ float shared_centroidY[NUM_THREADS_PER_BLOCK];

    if(lid == 0){
        for(int j = 0; j < myK; j++) {
            shared_centroidX[j] = myCentroidX[j];
            shared_centroidY[j] = myCentroidY[j];
        }
    }
    __syncthreads();
    
    if (id >= myN) return;

    float minDistancia = FP_INFINITE;
    int bestCluster = -1;
    //Calcular a distância entre todos os centroides.
    for(int j = 0; j < myK; j++) {
        //Distância euclidiana
        float xSub = (myPontoX[id] - shared_centroidX[j]);
        float ySub = (myPontoY[id] - shared_centroidY[j]);
        float distancia = xSub*xSub + ySub*ySub;
        if (distancia < minDistancia){
            minDistancia = distancia;
            bestCluster = j;
        }
    }

    //Atribuir o melhor cluster ao ponto
    myPontoCluster[id] = bestCluster;
}

__global__
void limpaSumatoriosKernel (int myK, float *mySumX, float *mySumY, int *myNumPontos){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= myK) return;

    mySumX[id] = 0;
    mySumY[id] = 0;
    myNumPontos[id] = 0;
}


__global__
void calculaSomatorioKernel (float *myPontoX, float *myPontoY, int myN,  int *myPontoCluster,
                             float *mySumX, float *mySumY, int *myNumPontos, int myK) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= myN) return;
    
    int clusterID = myPontoCluster[id];
    atomicAdd(&mySumX[clusterID], myPontoX[id]);
    atomicAdd(&mySumY[clusterID], myPontoY[id]);
    atomicAdd(&myNumPontos[clusterID], 1);
}

__global__
void calculaNovosCentroidsKernel (float *myCentroidX, float *myCentroidY, int myK, float *mySumX, float *mySumY, int *myNumPontos){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= myK) return;

    myCentroidX[id] = mySumX[id] / myNumPontos[id];
    myCentroidY[id] = mySumY[id] / myNumPontos[id];
}




int main(int argc,char** argv){
    if (argc != 3){
        printf("Número de argumentos incorretos\n");
        return 1;
    }

    int argNums[2] = {0,0};
    for(int i = 1;i<argc;i++){
        if (sscanf(argv[i],"%d",&argNums[i-1]) != 1){
            printf("Invalid argument '%s'. Only Numbers accepted\n",argv[i]);
            return 1;
        }
    }
    N = argNums[0];
    K = argNums[1];
    if (K > 1024 || N > 1000000 || K <= 0 || N <= 0){
        printf("Invalid argument values.\nValue of N must be between 1 and 1.000.000\nValue of K must be between 1 and 1024\n");
        return 1;
    }

    NUM_BLOCKS_FOR_POINTS = ((int) ceil((double) (N/NUM_THREADS_PER_BLOCK))) + 1;
    NUM_BLOCKS_FOR_CENTROIDS = 1;
    POINT_ARRAY_SIZE = N*sizeof(float);
    CENTROID_ARRAY_SIZE = K*sizeof(float);
    POINTCLUSTER_ARRAY_SIZE = N*sizeof(int);

    aloca();
    inicializa();
    initializeKernel();

    atribuiClusterKernel <<< NUM_THREADS_PER_BLOCK, NUM_BLOCKS_FOR_POINTS >>> (dpontoX, dpontoY, N, dcentroidX, dcentroidY, K, dpontoCluster);        
    checkCUDAError("kernel invocation");
    while(nIter < nMaxIter){
        //AJUSTA CENTROIDS
        limpaSumatoriosKernel <<< NUM_THREADS_PER_BLOCK, NUM_BLOCKS_FOR_CENTROIDS >>> (K, dsumX, dsumY, dnumPontos);
        calculaSomatorioKernel <<< NUM_THREADS_PER_BLOCK, NUM_BLOCKS_FOR_POINTS >>> (dpontoX, dpontoY, N, dpontoCluster, dsumX, dsumY, dnumPontos, K);
        calculaNovosCentroidsKernel <<< NUM_THREADS_PER_BLOCK, NUM_BLOCKS_FOR_CENTROIDS >>> (dcentroidX, dcentroidY, K, dsumX, dsumY, dnumPontos);    
        checkCUDAError("kernel invocation");

        atribuiClusterKernel <<< NUM_THREADS_PER_BLOCK, NUM_BLOCKS_FOR_POINTS >>> (dpontoX, dpontoY, N, dcentroidX, dcentroidY, K, dpontoCluster);
        checkCUDAError("kernel invocation");        

        nIter++;
    }

    //Copy results from device
    cudaMemcpy (centroidX,dcentroidX,CENTROID_ARRAY_SIZE,cudaMemcpyDeviceToHost);
    cudaMemcpy (centroidY,dcentroidY,CENTROID_ARRAY_SIZE,cudaMemcpyDeviceToHost);
    cudaMemcpy (pontoCluster,dpontoCluster,POINTCLUSTER_ARRAY_SIZE,cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy d->h");

    freeKernel();
    printFinalResult();
}

