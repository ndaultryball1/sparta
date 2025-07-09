#ifndef SPARTA_NN_MODEL_IMPL
#define SPARTA_NN_MODEL_IMPL

#include "torch/torch.h"

namespace SPARTA_NS {
  class NNCollideModelImpl : public torch::nn::Module {

    public:
      NNCollideModelImpl(int, int);

      std::tuple<torch::Tensor> forward( torch::Tensor) ;

    private:

      torch::nn::Linear R_fc1, R_fc2, R_fc_pi;
      torch::nn::Linear R_fc3, R_G1, R_G2;
  };
}

#endif