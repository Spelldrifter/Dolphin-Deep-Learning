#ifndef RBM_H
#define RBM_H
class Rbm{
  public:
    Rbm();
    Rbm(float momentum, float alpha, int visibleSize, int hiddenSize);
    Rbm(const Rbm &rbm);
    float computeCostAndGradient(float* &data, int batchSize);//data batchSize*visibleSize
    void updateWeight();
    void train(f