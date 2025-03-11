#ifndef SPARTA_NN_MODEL
#define SPARTA_NN_MODEL

#include "torch/torch.h"

namespace SPARTA_NS {
  class NNModel : public torch::nn::Module {
    public: 
      
      NNModel(int, int, int);
      torch::Tensor forward(torch::Tensor );
      void load_parameters(std::string );
      torch::nn::Linear fc1, fc2, fc3,  fc5, fc1_r, fc2_r, fc3_r,  fc5_r;
      torch::nn::Linear G1, G2, G1_r, G2_r;   
    private:
      std::vector<char> get_the_bytes(std::string);
    };
}
#endif
