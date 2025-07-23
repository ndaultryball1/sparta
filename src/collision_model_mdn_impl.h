#ifndef SPARTA_MDN_MODEL_IMPL
#define SPARTA_MDN_MODEL_IMPL

#include "torch/torch.h"

namespace SPARTA_NS {
  class MDNCollideModelImpl : public torch::nn::Module {

    public:
      MDNCollideModelImpl(int, int, int);

      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gen_params_chi( torch::Tensor) ;
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gen_params_R( torch::Tensor) ;
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gen_params_r( torch::Tensor) ;

    private:
      torch::nn::Linear chi_fc1, chi_fc2, chi_fc_pi, chi_fc_mu,  chi_fc_sigma;
      torch::nn::Linear chi_fc3,  chi_fc4,chi_G1, chi_G2, chi_G3;
      torch::nn::Linear R_fc1, R_fc2, R_fc_pi, R_fc_mu, R_fc_sigma;
      torch::nn::Linear R_fc3,  R_fc4, R_G1, R_G2, R_G3;
      torch::nn::Linear r_fc1, r_fc2, r_fc_pi, r_fc_mu, r_fc_sigma;
      torch::nn::Linear r_fc3,  r_fc4, r_G1, r_G2, r_G3;
  };
}

#endif