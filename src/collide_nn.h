#ifndef SPARTA_NN_MODEL
#define SPARTA_NN_MODEL

#include "torch/torch.h"

namespace SPARTA_NS {
  class NNModel : public torch::nn::Module {
    public: 
      
      NNModel(int, int, int);
      torch::Tensor forward(torch::Tensor );
      void load_parameters(std::string );
      torch::nn::Linear fc1, fc2, fc3, fc4, fc5;   
    private:
      std::vector<char> get_the_bytes(std::string);
    };
}
#endif
