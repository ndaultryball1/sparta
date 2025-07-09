#ifndef SPARTA_NN_COLLIDE_MODEL
#define SPARTA_NN_COLLIDE_MODEL

#include "torch/torch.h"
#include "collision_model.h"
#include "collision_model_nn_impl.h"

namespace SPARTA_NS {
  class NNCollideModel : public MLCollideModel {
    public: 
      
      NNCollideModel(class SPARTA *);
      virtual ~NNCollideModel();
      std::tuple<double, double, double> collide(double[]) override;
      void setup_model(int) override;

    private:
      NNCollideModelImpl* derived_implementation;
      NNCollideModelImpl* get_implementation() override {return derived_implementation;};

      void set_implementation(NNCollideModelImpl* x){ derived_implementation = x;};

      torch::Tensor forward(torch::Tensor , torch::Tensor , torch::Tensor);
      void update_LR( int ) override;

      torch::Tensor get_loss(torch::Tensor, torch::Tensor ) override;

      void optimizer_step() override;
      void broadcast_weights() override;
      void save_weights(int) override;

      void read_params() override;

      struct NNParams { 
        int width;

        std::string model_loc;
      };

      NNParams NN_params;
    };
}
#endif