#include "collide_mdn_multi.h"
#include "math_const.h"
#include "torch/torch.h"
#include <fstream>

using namespace SPARTA_NS;
using namespace MathConst;
using namespace torch::indexing;

#define oneDivSqrtTwoPI  1.0 / sqrt(2.0*MY_PI) ;

MDNModelMulti::MDNModelMulti(int n_inputs, int n_hidden, int n_gaussians):
    fc1(n_inputs, n_hidden ),
    fc2( n_hidden, n_hidden ),
    fc_pi(n_hidden, n_gaussians ),
    fc_mu( n_hidden, 3*n_gaussians  ),
    fc_U(n_hidden, 6*n_gaussians  )
{
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc_mu", fc_mu);
    register_module("fc_pi", fc_pi);
    register_module("fc_U", fc_U);

    num_gaussians = n_gaussians;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MDNModelMulti::gen_params( torch::Tensor x) {
        torch::Tensor H1 = torch::tanh( fc1(x));
        torch::Tensor H2 = torch::tanh( fc2(H1));

        torch::Tensor pi = torch::softmax(fc_pi(H2), -1);
        torch::Tensor U = fc_U(H2).reshape({-1,num_gaussians, 6});
        torch::Tensor mu = fc_mu(H2).reshape({-1,num_gaussians, 3});
        return {pi, U, mu};
}

torch::Tensor MDNModelMulti::draw_gumbel(double loc, double scale, torch::Tensor example){
    torch::Tensor p = torch::rand_like(example);
    torch::Tensor z = loc * torch::ones_like(example) - scale * torch::log(-torch::log(p));
    return(z);
}

torch::Tensor MDNModelMulti::fill_cholesky(torch::Tensor U_entries, int batch_size){

    auto options = torch::TensorOptions().dtype(torch::kBool);
    torch::Tensor mask_upper = torch::triu(torch::ones({dim, dim}), 1) == 1;
    torch::Tensor mask_diagonal = torch::eye(dim, dim, options);

    torch::Tensor U = torch::zeros({batch_size, num_gaussians, dim, dim} );

    // Fill everything above the diagonal as is
    U.index_put_({mask_upper.expand({batch_size,num_gaussians,-1,-1})} , U_entries.index({Slice(), Slice(), Slice(dim, None)}).reshape(-1) );
    // Diagonal entries must be positive
    U.index_put_({mask_diagonal.expand({batch_size,num_gaussians,-1,-1})}, U_entries.index({Slice(),Slice(),Slice(None,dim)}).exp().reshape(-1)) ;
    return(U);
}

torch::Tensor MDNModelMulti::neg_log_likelihood( torch::Tensor pi, torch::Tensor U, torch::Tensor mu, torch::Tensor y ){
    int dim = 3;
    int batch_size = pi.sizes()[0];
    torch::Tensor y_hat = torch::log(y/(1-y));

    torch::Tensor j = U.index({Slice(), Slice(), Slice(None,dim)}).sum(-1);
    torch::Tensor U_bar = fill_cholesky(U, batch_size);
   
    torch::Tensor z = (U_bar* (y_hat.index({Slice(), None, Slice()}) - mu).index({Slice(), Slice(), None, Slice()})).sum(-1);
    return( (-((-0.5 * (z*z).sum(-1) + j).exp() * MY_PI).sum(-1).log()).mean() );
}

torch::Tensor MDNModelMulti::inv(torch::Tensor m){
    // Invert 3x3 tensor

    double det = (m.index({0, 0}) * (m.index({1, 1}) * m.index({2, 2}) - m.index({2, 1}) * m.index({1, 2})) -
             m.index({0, 1}) * (m.index({1, 0}) * m.index({2, 2}) - m.index({1, 2}) * m.index({2, 0})) +
             m.index({0, 2}) * (m.index({1, 0}) * m.index({2, 1}) - m.index({1, 1}) * m.index({2, 0})))[0].item<double>();

    double invdet = 1 / det;

    torch::Tensor minv = torch::zeros_like(m); // inverse of matrix m
    minv.index_put_({0, 0}, (m.index({1, 1}) * m.index({2, 2}) - m.index({2, 1}) * m.index({1, 2})) * invdet);
    minv.index_put_({0, 1}, (m.index({0, 2}) * m.index({2, 1}) - m.index({0, 1}) * m.index({2, 2})) * invdet);
    minv.index_put_({0, 2}, (m.index({0, 1}) * m.index({1, 2}) - m.index({0, 2}) * m.index({1, 1})) * invdet);
    minv.index_put_({1, 0}, (m.index({1, 2}) * m.index({2, 0}) - m.index({1, 0}) * m.index({2, 2})) * invdet);
    minv.index_put_({1, 1}, (m.index({0, 0}) * m.index({2, 2}) - m.index({0, 2}) * m.index({2, 0})) * invdet);
    minv.index_put_({1, 2}, (m.index({1, 0}) * m.index({0, 2}) - m.index({0, 0}) * m.index({1, 2})) * invdet);
    minv.index_put_({2, 0}, (m.index({1, 0}) * m.index({2, 1}) - m.index({2, 0}) * m.index({1, 1})) * invdet);
    minv.index_put_({2, 1}, (m.index({2, 0}) * m.index({0, 1}) - m.index({0, 0}) * m.index({2, 1})) * invdet);
    minv.index_put_({2, 2}, (m.index({0, 0}) * m.index({1, 1}) - m.index({1, 0}) * m.index({0, 1})) * invdet);

    return(minv);
}

torch::Tensor MDNModelMulti::forward(torch::Tensor input){
    int dim = 3;
    auto [pi, U, mu_test] = gen_params(input);
    torch::Tensor gumbel = draw_gumbel(0,1, pi);
    torch::Tensor k = torch::argmax( torch::log(pi) + gumbel, -1);
    torch::Tensor rn = torch::randn({1,dim});
    torch::Tensor U_bar = fill_cholesky(U, 1);
    torch::Tensor L = inv(U_bar); // Also consider doing a linear solve to get the output of the below left multiplication.

    torch::Tensor sampled = (L.index({k}) *rn.index({Slice(), None, Slice()})).sum(-1)  + mu_test.index({k});
    sampled = torch::sigmoid(sampled);

    return(sampled);
}

// Below needed to load models saved from python.

std::vector<char> MDNModelMulti::get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

void MDNModelMulti::load_parameters(std::string pt_pth) {
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
