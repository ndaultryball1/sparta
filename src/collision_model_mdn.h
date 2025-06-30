#ifndef SPARTA_MDN_COLLIDE_MODEL
#define SPARTA_MDN_COLLIDE_MODEL

#include "torch/torch.h"
#include "collision_model.h"
#include "collision_model_mdn_impl.h"

namespace SPARTA_NS {
  class MDNCollideModel : public MLCollideModel {
    public: 
      
      MDNCollideModel(class SPARTA *);
      virtual ~MDNCollideModel();
      std::tuple<double, double, double> collide(double[]) override;
      void setup_model(int) override;

    private:
      MDNCollideModelImpl* derived_implementation;
      MDNCollideModelImpl* get_implementation() override {return derived_implementation;};

      void set_implementation(MDNCollideModelImpl* x){ derived_implementation = x;};

      torch::Tensor draw_gumbel(double , double , torch::Tensor );

      torch::Tensor forward(torch::Tensor , torch::Tensor , torch::Tensor);
      void update_LR( int ) override;

      torch::Tensor get_loss(torch::Tensor, torch::Tensor ) override;

      void optimizer_step() override;
      void broadcast_weights() override;
      void save_weights(int) override;

      void read_params() override;

      struct MDNParams { // Hyperparameters for architecture of 3xMDNs with same width
        int width;
        int gaussians;

        std::string model_loc;
      };

      MDNParams mdn_params;

      torch::Tensor neg_log_likelihood( torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor );

      torch::Tensor gaussian_distribution( torch::Tensor, torch::Tensor, torch::Tensor);
    };
}
#endif