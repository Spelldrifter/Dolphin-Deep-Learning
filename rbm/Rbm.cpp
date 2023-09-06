
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