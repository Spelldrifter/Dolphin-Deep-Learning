
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