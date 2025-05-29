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
  chi_fc1(n_inputs, n_hidden ), chi_fc2( n_hidden, n_hidden ), chi_fc_pi(n_hidden, n_gaussians ), chi_fc_mu(n_hidden, n_gaussians ),  chi_fc_sigma(n_hidden, n_gaussians ),
  R_fc1(n_inputs+1, n_hidden ), R_fc2( n_hidden, n_hidden ), R_fc_pi(n_hidden, n_gaussians ), R_fc_mu(n_hidden, n_gaussians ), R_fc_sigma(n_hidden, n_gaussians ),
  r_fc1(n_inputs+1, n_hidden ), r_fc2( n_hidden, n_hidden ), r_fc_pi(n_hidden, n_gaussians ), r_fc_mu(n_hidden, n_gaussians ), r_fc_sigma(n_hidden, n_gaussians )
{
  register_module("chi_fc1", chi_fc1);
  register_module("chi_fc2", chi_fc2);
  register_module("chi_fc_mu", chi_fc_mu);
  register_module("chi_fc_pi", chi_fc_pi);
  register_module("chi_fc_sigma", chi_fc_sigma);

  register_module("R_fc1", R_fc1);
  register_module("R_fc2", R_fc2);
  register_module("R_fc_mu", R_fc_mu);
  register_module("R_fc_pi", R_fc_pi);
  register_module("R_fc_sigma", R_fc_sigma);

  register_module("r_fc1", r_fc1);
  register_module("r_fc2", r_fc2);
  register_module("r_fc_mu", r_fc_mu);
  register_module("r_fc_pi", r_fc_pi);
  register_module("r_fc_sigma", r_fc_sigma);

}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MDNCollideModelImpl::gen_params_chi( torch::Tensor x) {
        torch::Tensor H1 = torch::tanh( chi_fc1(x));
        torch::Tensor H2 = torch::tanh( chi_fc2(H1));

        torch::Tensor pi = torch::softmax(chi_fc_pi(H2), -1);
        torch::Tensor sigma = torch::exp(chi_fc_sigma(H2));
        torch::Tensor mu = chi_fc_mu(H2);
        return {pi, sigma, mu};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MDNCollideModelImpl::gen_params_R( torch::Tensor x) {
        torch::Tensor H1 = torch::tanh( R_fc1(x));
        torch::Tensor H2 = torch::tanh( R_fc2(H1));

        torch::Tensor pi = torch::softmax(R_fc_pi(H2), -1);
        torch::Tensor sigma = torch::exp(R_fc_sigma(H2));
        torch::Tensor mu = R_fc_mu(H2);
        return {pi, sigma, mu};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MDNCollideModelImpl::gen_params_r( torch::Tensor x) {
        torch::Tensor H1 = torch::tanh( r_fc1(x));
        torch::Tensor H2 = torch::tanh( r_fc2(H1));

        torch::Tensor pi = torch::softmax(r_fc_pi(H2), -1);
        torch::Tensor sigma = torch::exp(r_fc_sigma(H2));
        torch::Tensor mu = r_fc_mu(H2);
        return {pi, sigma, mu};
}