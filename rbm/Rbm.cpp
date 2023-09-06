
#include "mkl.h"
#include "stdlib.h"
#include <iostream>
#include <math.h>
#include <string.h>
#include "Rbm.h"
#include "consts.h"

using namespace std;
Rbm::Rbm(float momentum, float alpha, int visibleSize, int hiddenSize){
  this->momentum = momentum;
  this->alpha = alpha;
  this->visibleSize = visibleSize;
  this->hiddenSize = hiddenSize;
  
  W = (float*)mkl_malloc(sizeof(float) * hiddenSize * visibleSize, 64);
  vW = (float*)mkl_malloc(sizeof(float) * hiddenSize * visibleSize, 64);
  memset(W, 0, sizeof(float) * hiddenSize * visibleSize);
  b = (float*)mkl_malloc(sizeof(float) * visibleSize, 64);
  vb = (float*)mkl_malloc(sizeof(float) * visibleSize, 64);
  memset(b, 0, sizeof(float) * visibleSize);
  memset(vb, 0, sizeof(float) * visibleSize);
  c = (float*)mkl_malloc(sizeof(float) * hiddenSize, 64);
  vc = (float*)mkl_malloc(sizeof(float) * hiddenSize, 64);
  memset(c, 0, sizeof(float) * hiddenSize);
  memset(vc, 0, sizeof(float) * hiddenSize);
  random = (float*)mkl_malloc(sizeof(float) * (hiddenSize >= visibleSize ? hiddenSize:visibleSize), 64);
  int max = hiddenSize >= visibleSize ? hiddenSize : visibleSize;
  for(int i = 0; i < max; i++){
    random[i] = rand() / RAND_MAX;
  }
}

Rbm::~Rbm(){
  mkl_free(W);
  mkl_free(vW);
  mkl_free(b);
  mkl_free(vb);
  mkl_free(c);
  mkl_free(vc);
  mkl_free(random);
}
float Rbm::computeCostAndGradient(float* &data, int batchSize){
  float error = 0.0;  //v1 = data
  //h1 = sigmrnd(v1*W'+c) batchSize * hiddenSize
  /*float* h1 = (float*)mkl_malloc(sizeof(float) * batchSize * hiddenSize, 64);
  memset(h1, 0, sizeof(float) * batchSize * hiddenSize);*/
  //printf("h1:%p\n",h1);
  //printf("before the first matrix\n");
  //printf("1\n");
  /*cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		batchSize, hiddenSize, visibleSize,
		1.0, data, visibleSize,
		W, visibleSize, 0.0, 
		h1, hiddenSize);*/
  for(int i = 0; i < batchSize; i++){
    for(int j = 0; j < hiddenSize; j++){
        h1[i * hiddenSize + j] = 0;
        for(int k = 0; k < visibleSize; k++){
            h1[i * hiddenSize + j] += data[i * visibleSize + k] * W[j * visibleSize + k];
        }
    }
  }
  //printf("2\n");
  //#pragma omp parallel for num_threads(NUM_THREADS/8)
  //#pragma ivdep
  for(int i = 0; i < batchSize * hiddenSize; i++){
    h1[i] = 1 / (1 + exp(-1 * (h1[i] + c[i % hiddenSize])));
    if(h1[i] > random[i % hiddenSize])
      h1[i] = 1;
    else
      h1[i] = 0;
  }
//  printf("3\n");
  //v2 = sigmrnd(h1*W+b) batchSize * visibleSize
  /*float* v2 = (float*)mkl_malloc(sizeof(float) * batchSize * visibleSize, 64);
  memset(v2, 0, sizeof(float) * batchSize * visibleSize);*/
  //printf("v2:%p\n",v2);
  /*cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		batchSize, visibleSize, hiddenSize,
		1.0, h1, hiddenSize,
		W, visibleSize, 0.0, 
		v2, visibleSize);*/
  for(int i = 0; i < batchSize; i++){
    for(int j = 0; j < visibleSize; j++){
        v2[i * visibleSize + j] = 0.0;
        for(int k = 0; k < hiddenSize; k++){
            v2[i * visibleSize + j] += h1[i * hiddenSize + k] * W[k * visibleSize + j];
        }
    }
  }
  //printf("4\n");
  //#pragma omp parallel for num_threads(NUM_THREADS/8)
 // #pragma ivdep
  for(int i = 0; i < batchSize * visibleSize; i++){
    v2[i] = 1 / (1 + exp(-1 * (v2[i] + b[i % visibleSize])));
    if(v2[i] > random[i % visibleSize])
      v2[i] = 1;
    else
      v2[i] = 0;
  }

  //printf("%d",2);
  //h2 = sigm(v2*W'+c)
  /*float* h2 = (float*)mkl_malloc(sizeof(float) * batchSize * hiddenSize, 64);
  memset(h2, 0, sizeof(float) * batchSize * hiddenSize);*/
  //printf("h2:%p\n",h2);
  /*cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		batchSize, hiddenSize, visibleSize,
		1.0, v2, visibleSize,
		W, visibleSize, 0.0, 
		h2, hiddenSize);*/
  for(int i = 0;i < batchSize; i++){
    for(int j = 0; j < hiddenSize; j++){