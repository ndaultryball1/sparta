#ifndef SPARTA_MDN_MODEL_MULTI
#define SPARTA_MDN_MODEL_MULTI

#include "torch/torch.h"

namespace SPARTA_NS {
  class MDNModelMulti : public torch::nn::Module {
    public: 
      
      MDNModelMulti(int, int, int);
      torch::Tensor forward(torch::Tensor );
      
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gen_params(torch::Tensor);
      void load_parameters(std::string );

      torch::Tensor neg_log_likelihood( torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor );
      
    private:
      torch::nn::Linear fc1, fc2, fc_pi, fc_mu,  fc_U;
      torch::Tensor draw_gumbel(double , double , torch::Tensor );
      std::vector<char> get_the_bytes(std::string);
      torch::Tensor fill_cholesky(torch::Tensor,  int);
      torch::Tensor inv(torch::Tensor);

      int num_gaussians;
      int dim = 3;
    };
}
#endif
