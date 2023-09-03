SRC_FILES=Autoencoder.cpp  main.cpp  SoftMax.cpp  StackedAutoencoder.cpp matrixm.cpp
HEADER_FILES=Autoencoder.h  matrixm.h  SoftMax.h  StackedAutoencoder.h consts.h
CC=icc
CXX=icpc
FLAGS=-g -vec-report -mkl -O2 -fopenmp -openmp-report=2 -lrt

all: stackedAutoEncoder stackedAutoEncoder.mic

stackedAutoEncoder: ${SRC_FILES} ${HEADER_FILES} Makefile
	${CXX} ${SRC_FILES} ${FLAGS} -o stackedAutoEncoder
	
stackedAutoEncoder.mic: ${SRC_FILES} ${HEADER_FILES} Makefile
	${CXX} ${SRC_FILES} ${FLAGS} -mmic -o stackedAutoEncoder.mic
run: stackedAutoEncoder
	time ./sta