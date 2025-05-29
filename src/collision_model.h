#ifndef SPARTA_ML_COLLIDE_MODEL
#define SPARTA_ML_COLLIDE_MODEL

#include "torch/torch.h"
#include "pointers.h"

enum{NO,START,ALL, OFFLINE};

namespace SPARTA_NS {
  class MLCollideModel : protected Pointers {
    public: 
      
      MLCollideModel(class SPARTA *);
      ~MLCollideModel() override;
      virtual std::tuple<double, double, double> collide(double[] )=0;
      void load_weights(std::string );

      virtual void setup_model(int)=0;
      void train(int,int);
      int requires_data();

      struct TrainData {
        std::vector< double > features; 
        std::vector< double > outputs;
        int num_features;
        int num_outputs;
      };

      struct TrainParams { // Hyperparameters for training of a neural network model
        int width;
        int train_every;
        int train_max; // When to stop training
        int epochs;
        int len_data;
        double LR;
        double A; 
        double B; 
        double C; // Parameters for learning rate decay
        int batch_size;
        double e_ref;
        double b_ref;
      };

      TrainParams train_params; // This should also be an array later.
      TrainData training_data;

    protected:
      int wordparse(int, char *, char **);
      void read_train_params();

      std::shared_ptr<torch::optim::Adam> optimizer;
      std::shared_ptr<torch::nn::Module> implementation; 


    private:
      std::vector<char> get_the_bytes(std::string);
      int train_this_step(int,int);
      torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
      int total_epochs;
      void save_to_disk(torch::Tensor, torch::Tensor, int);

      virtual void update_LR(int )=0;

      virtual torch::Tensor get_loss(torch::Tensor, torch::Tensor )=0;

      virtual void optimizer_step()=0;
      virtual void broadcast_weights()=0;
      virtual void save_weights()=0;

      virtual void read_params()=0;

      

    };
}
#endif