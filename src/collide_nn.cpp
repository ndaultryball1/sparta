#include "collide_nn.h"
#include "math_const.h"
#include "torch/torch.h"
#include <fstream>

using namespace SPARTA_NS;
using namespace MathConst;

NNModel::NNModel(int N, int H, int O):
    fc1(N, H ),
    fc2( H, H ),
    fc3( H, O )
{
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

torch::Tensor NNModel::forward(torch::Tensor input){
    torch::Tensor inter = fc1( input );
    torch::Tensor H1 = torch::relu( inter );
    torch::Tensor H2 = torch::relu( fc2(H1) );
    return( MY_PI * torch::sigmoid( fc3(H2) ) );
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