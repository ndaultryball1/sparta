#include "collision_model_nn_impl.h"
#include "math_const.h"
#include "math.h"
#include "random_knuth.h"
#include "mixture.h"
#include "string.h"

#include "stdlib.h"
#include "error.h"
#include "update.h"

#define NUM_OUT 3

using namespace SPARTA_NS;
using namespace MathConst;


NNCollideModelImpl::NNCollideModelImpl(int n_inputs, int n_hidden) : 

  R_fc1(n_inputs+1, n_hidden ), R_fc2( n_hidden, n_hidden ), R_fc3( n_hidden, n_hidden ),
  R_G1(n_inputs+1, n_hidden), R_G2(n_inputs+1, n_hidden), 
  R_fc_pi(n_hidden, NUM_OUT )
{
  register_module("R_fc1", R_fc1);
  register_module("R_fc2", R_fc2);
  register_module("R_fc3", R_fc3);
  register_module("R_G1", R_G1);
  register_module("R_G2", R_G2);
  register_module("R_fc_pi", R_fc_pi);
}

std::tuple<torch::Tensor> NNCollideModelImpl::forward( torch::Tensor x) {
        torch::Tensor H1 = torch::tanh(R_fc1(x));
        torch::Tensor H2 = torch::tanh(R_fc2(H1));

        torch::Tensor H3 = torch::tanh(R_G1(x)) * H2;
        torch::Tensor H4 = torch::tanh(R_fc3(H3));

        torch::Tensor H5 = torch::tanh(R_G2(x)) * H4;

        torch::Tensor pi = torch::sigmoid(R_fc_pi(H5));
        return {pi};
}
