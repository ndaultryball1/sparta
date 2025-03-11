#include "collide_nn.h"
#include "math_const.h"
#include "torch/torch.h"
#include <fstream>

using namespace SPARTA_NS;
using namespace MathConst;

NNModel::NNModel(int N, int H, int O):
    fc1(N, H ),
    fc2( H, H ),
    fc3( H,H ),
    fc5( H, 2 ),
    G1(N,H),
    G2(N,H),
    fc1_r(N, H ),
    fc2_r( H, H ),
    fc3_r( H,H ),
    fc5_r( H, 1 ),
    G1_r(N,H),
    G2_r(N,H)
{
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
    register_module("fc5", fc5);

    register_module("G1", G1);
    register_module("G2", G2);

    register_module("fc1_r", fc1_r);
    register_module("fc2_r", fc2_r);
    register_module("fc3_r", fc3_r);
    register_module("fc5_r", fc5_r);

    register_module("G1_r", G1_r);
    register_module("G2_r", G2_r);

}

torch::Tensor NNModel::forward(torch::Tensor input){
    torch::Tensor H1 = torch::tanh(fc1(input));
    torch::Tensor H2 = torch::tanh(fc2(H1));

    torch::Tensor H3 = torch::tanh(G1(input)) * H2;
    torch::Tensor H4 = torch::tanh(fc3(H3));

    torch::Tensor H5 = torch::tanh(G2(input)) * H4;
    torch::Tensor chi_R= torch::sigmoid(fc5(H5));

    torch::Tensor H1_r = torch::tanh(fc1_r(input));
    torch::Tensor H2_r = torch::tanh(fc2_r(H1_r));

    torch::Tensor H3_r = torch::tanh(G1_r(input)) * H2_r;
    torch::Tensor H4_r = torch::tanh(fc3_r(H3_r));

    torch::Tensor H5_r = torch::tanh(G2_r(input)) * H4_r;
    torch::Tensor r = torch::sigmoid(fc5_r(H5_r));
    torch::Tensor res = torch::cat( {chi_R,r},-1 );
    return(res);
}

// Below needed to load models saved from python.

std::vector<char> NNModel::get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

void NNModel::load_parameters(std::string pt_pth) {
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
