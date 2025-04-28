#ifndef SPARTA_MDN_MODEL
#define SPARTA_MDN_MODEL

#include "torch/torch.h"

namespace SPARTA_NS {
  class MDNModel : public torch::nn::Module {
    public: 
      
      MDNModel(int, int, int);
      torch::Tensor forward(torch::Tensor );
      
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gen_params(torch::Tensor);
      void load_parameters(std::string );

      torch::Tensor neg_log_likelihood( torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor );
      
    private:
      torch::nn::Linear fc1, fc2, fc_pi, fc_mu,  fc_sigma;
      torch::Tensor draw_gumbel(double , double , torch::Tensor );
      std::vector<char> get_the_bytes(std::string);
      torch::Tensor gaussian_distribution( torch::Tensor, torch::Tensor, torch::Tensor);
    };
}
#endif
