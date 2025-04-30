#include "collide_mdn.h"
#include "math_const.h"
#include "torch/torch.h"
#include <fstream>

using namespace SPARTA_NS;
using namespace MathConst;

#define oneDivSqrtTwoPI  1.0 / sqrt(2.0*MY_PI) ;

MDNModel::MDNModel(int n_inputs, int n_hidden, int n_gaussians):
    fc1(n_inputs, n_hidden ),
    fc2( n_hidden, n_hidden ),
    fc_pi(n_hidden, n_gaussians ),
    fc_mu( n_hidden, n_gaussians  ),
    fc_sigma(n_hidden, n_gaussians  )
{
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc_mu", fc_mu);
    register_module("fc_pi", fc_pi);
    register_module("fc_sigma", fc_sigma);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MDNModel::gen_params( torch::Tensor x) {
        torch::Tensor H1 = torch::tanh( fc1(x));
        torch::Tensor H2 = torch::tanh( fc2(H1));

        torch::Tensor pi = torch::softmax(fc_pi(H2), -1);
        torch::Tensor sigma = torch::exp(fc_sigma(H2));
        torch::Tensor mu = fc_mu(H2);
        return {pi, sigma, mu};
}

torch::Tensor MDNModel::draw_gumbel(double loc, double scale, torch::Tensor example){
    torch::Tensor p = torch::rand_like(example);
    torch::Tensor z = loc * torch::ones_like(example) - scale * torch::log(-torch::log(p));
    return(z);
}

torch::Tensor MDNModel::gaussian_distribution( torch::Tensor y, torch::Tensor mu, torch::Tensor sigma){
    torch::Tensor result = (y.index({torch::indexing::Slice(torch::indexing::None),torch::indexing::None}) - mu) * torch::reciprocal(sigma);
    result = -0.5 * (result * result);
    return(torch::exp(result) * torch::reciprocal(sigma)) * oneDivSqrtTwoPI;
}


torch::Tensor MDNModel::neg_log_likelihood( torch::Tensor pi, torch::Tensor sigma, torch::Tensor mu, torch::Tensor y ){
    torch::Tensor result = gaussian_distribution(y, mu, sigma) * MY_PI;
    result = torch::mean(-torch::log(result.sum(1)));
    return( result );
}


torch::Tensor MDNModel::forward(torch::Tensor input){

    auto [pi_weights, sigma, mu] = gen_params(input);
    torch::Tensor gumbel = draw_gumbel(0,1, pi_weights);
    torch::Tensor k = torch::argmax( torch::log(pi_weights) + gumbel, -1);
    torch::Tensor rn = torch::randn(1);
    torch::Tensor out = rn * sigma.index({k} ) + mu.index({ k});
    return(out);
}

// Below needed to load models saved from python.

std::vector<char> MDNModel::get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

void MDNModel::load_parameters(std::string pt_pth) {
  std::vector<char> f = this->get_the_bytes(pt_pth);
  c10::Dict<c10::IValue, c10::IValue> weights = torch::pickle_load(f).toGenericDict();

  const torch::OrderedDict<std::string, at::Tensor>& model_params = this->named_parameters();
  std::vector<std::string> param_names;
  for (auto const& w : model_params) {
    param_names.push_back(w.key());
  }

  torch::NoGradGuard no_grad;
  for (auto const& w : weights) {
      std::string name = w.key().toStringRef();
      at::Tensor param = w.value().toTensor();

      if (std::find(param_names.begin(), param_names.end(), name) != param_names.end()){
        model_params.find(name)->copy_(param);
      } else {
        std::cout << name << " does not exist among model parameters." << std::endl;
      };

  }
}
