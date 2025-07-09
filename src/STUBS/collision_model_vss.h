#ifndef SPARTA_VSS_COLLIDE_MODEL
#define SPARTA_VSS_COLLIDE_MODEL

#include "torch/torch.h"
#include "collision_model.h"

namespace SPARTA_NS {
  class VSSCollideModel : public MLCollideModel {
    public: 
      
      VSSCollideModel(class SPARTA *);
      virtual ~VSSCollideModel();
      std::tuple<double, double, double> collide(double[]) override;
      void setup_model(int) override;

    private:

      torch::Tensor forward(torch::Tensor , torch::Tensor , torch::Tensor);
      void update_LR( int ) override;

      torch::Tensor get_loss(torch::Tensor, torch::Tensor ) override;

      void optimizer_step() override;
      void broadcast_weights() override;
      void save_weights(int) override;

      void read_params() override;

      struct VSSParams { // Hyperparameters for architecture of 3xVSSs with same width
        
      };

      VSSParams VSS_params;

      torch::Tensor neg_log_likelihood( torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor );

      torch::Tensor gaussian_distribution( torch::Tensor, torch::Tensor, torch::Tensor);
    };
}
#endif