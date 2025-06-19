#include "collision_model_mdn_impl.h"
#include "math_const.h"
#include "math.h"
#include "random_knuth.h"
#include "mixture.h"
#include "string.h"

#include "stdlib.h"
#include "error.h"
#include "update.h"


using namespace SPARTA_NS;
using namespace MathConst;


MDNCollideModelImpl::MDNCollideModelImpl(int n_inputs, int n_hidden, int n_gaussians) : 
  chi_fc1(n_inputs, n_hidden ), chi_fc2( n_hidden, n_hidden ),chi_fc3( n_hidden, n_hidden ),
  chi_G1(n_inputs, n_hidden), chi_G2(n_inputs, n_hidden), 
  chi_fc_pi(n_hidden, n_gaussians ), chi_fc_mu(n_hidden, n_gaussians ),  chi_fc_sigma(n_hidden, n_gaussians ),
  R_fc1(n_inputs+1, n_hidden ), R_fc2( n_hidden, n_hidden ), R_fc3( n_hidden, n_hidden ),
  R_G1(n_inputs+1, n_hidden), R_G2(n_inputs+1, n_hidden), 
  R_fc_pi(n_hidden, n_gaussians ), R_fc_mu(n_hidden, n_gaussians ), R_fc_sigma(n_hidden, n_gaussians ),
  r_fc1(n_inputs+1, n_hidden ), r_fc2( n_hidden, n_hidden ), r_fc3( n_hidden, n_hidden ),
  r_G1(n_inputs+1, n_hidden), r_G2(n_inputs+1, n_hidden), 
  r_fc_pi(n_hidden, n_gaussians ), r_fc_mu(n_hidden, n_gaussians ), r_fc_sigma(n_hidden, n_gaussians )
{
  register_module("chi_fc1", chi_fc1);
  register_module("chi_fc2", chi_fc2);
  register_module("chi_fc3", chi_fc3);
  register_module("chi_G1", chi_G1);
  register_module("chi_G2", chi_G2);
  register_module("chi_fc_mu", chi_fc_mu);
  register_module("chi_fc_pi", chi_fc_pi);
  register_module("chi_fc_sigma", chi_fc_sigma);

  register_module("R_fc1", R_fc1);
  register_module("R_fc2", R_fc2);
  register_module("R_fc3", R_fc3);
  register_module("R_G1", R_G1);
  register_module("R_G2", R_G2);
  register_module("R_fc_mu", R_fc_mu);
  register_module("R_fc_pi", R_fc_pi);
  register_module("R_fc_sigma", R_fc_sigma);

  register_module("r_fc1", r_fc1);
  register_module("r_fc2", r_fc2);
  register_module("r_fc3", r_fc3);
  register_module("r_G1", r_G1);
  register_module("r_G2", r_G2);
  register_module("r_fc_mu", r_fc_mu);
  register_module("r_fc_pi", r_fc_pi);
  register_module("r_fc_sigma", r_fc_sigma);

}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MDNCollideModelImpl::gen_params_chi( torch::Tensor x) {
        
        torch::Tensor H1 = torch::tanh(chi_fc1(x));
        torch::Tensor H2 = torch::tanh(chi_fc2(H1));

        torch::Tensor H3 = torch::tanh(chi_G1(x)) * H2;
        torch::Tensor H4 = torch::tanh(chi_fc3(H3));

        torch::Tensor H5 = torch::tanh(chi_G2(x)) * H4;


        torch::Tensor pi = torch::softmax(chi_fc_pi(H5), -1);
        torch::Tensor sigma = torch::exp(chi_fc_sigma(H5));
        torch::Tensor mu = chi_fc_mu(H5);
        return {pi, sigma, mu};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MDNCollideModelImpl::gen_params_R( torch::Tensor x) {
        torch::Tensor H1 = torch::tanh(R_fc1(x));
        torch::Tensor H2 = torch::tanh(R_fc2(H1));

        torch::Tensor H3 = torch::tanh(R_G1(x)) * H2;
        torch::Tensor H4 = torch::tanh(R_fc3(H3));

        torch::Tensor H5 = torch::tanh(R_G2(x)) * H4;

        torch::Tensor pi = torch::softmax(R_fc_pi(H5), -1);
        torch::Tensor sigma = torch::exp(R_fc_sigma(H5));
        torch::Tensor mu = R_fc_mu(H5);
        return {pi, sigma, mu};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MDNCollideModelImpl::gen_params_r( torch::Tensor x) {
        torch::Tensor H1 = torch::tanh(r_fc1(x));
        torch::Tensor H2 = torch::tanh(r_fc2(H1));

        torch::Tensor H3 = torch::tanh(r_G1(x)) * H2;
        torch::Tensor H4 = torch::tanh(r_fc3(H3));

        torch::Tensor H5 = torch::tanh(r_G2(x)) * H4;

        torch::Tensor pi = torch::softmax(r_fc_pi(H5), -1);
        torch::Tensor sigma = torch::exp(r_fc_sigma(H5));
        torch::Tensor mu = r_fc_mu(H5);
        return {pi, sigma, mu};
}