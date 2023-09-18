
#include <iostream>
#include "mkl.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include <fstream>
#include <sstream>
#include <pthread.h>
#include "Rbm.h"

using namespace std;
char tstr[1000000];
float* chunk;
pthread_mutex_t mutex[10];
pthread_cond_t cond[10];
bool full[10];
int iter;
extern void readData(float* &data, string filename);

extern void* trainingThread(void*);
extern void* loadingThread(void*);
extern string itos(int& i);
/*
 * 
 */
int main(int argc, char** argv) { 
    float *data;
    data = (float*)mkl_malloc( 576 * 100000 * sizeof(float), 64 );
    memset(data, 0, sizeof(float) * 576 * 100000);
    Rbm* rbm = new Rbm(0.1, 1, 576, 1024);
    string path = "./DataSet/";
    readData(data, "./DataSet/576.txt");
    cerr << "finish reading data" << endl;
    float* p;
    for(int i = 0; i < 500; i++){
	p = &data[(i%50) * 200 * 576];
	//printf("hereh\n");
        rbm->train(p, 1, 200);
	printf("iter:%d\n",i);
    }
    //rbm->train(data, 20, 10000);
    mkl_free(data);
    delete rbm;
}