#ifndef RBM_H
#define RBM_H
class Rbm{
  public:
    Rbm();
    Rbm(float momentum, float alpha, int visibleSize, int hiddenSize);
    Rbm(const Rbm &rbm);
    float computeCostAndGradient(float* &data, int batchSize);//data batchSize*visibleSize
    void updateWeight();
    void train(float* &data, int iter, int batchSize);
    void sigm(float* &data, int batchSize);
    void sigmrnd(float* &data, int batchSize);
    virtual ~Rbm();
    
    int visibleSize;
    int hiddenSize;
  