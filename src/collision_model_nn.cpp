#include "collision_model_nn.h"
#include "math_const.h"
#include "comm.h"
#include "math.h"
#include "random_knuth.h"
#include "mixture.h"
#include "string.h"

#include "stdlib.h"
#include "error.h"
#include "update.h"

#include "torch/torch.h"

#define MAXLINE 1024
#define oneDivSqrtTwoPI  1.0 / sqrt(2.0*MY_PI) ;

using namespace SPARTA_NS;
using namespace MathConst;
using namespace torch::indexing;

NNCollideModel::NNCollideModel(SPARTA *sparta) :
  MLCollideModel(sparta)
{
}

NNCollideModel::~NNCollideModel(){}


std::tuple<double, double, double> NNCollideModel::collide(double input_data[]){

  double chi, R, r;
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor inputs = torch::from_blob(input_data, {10}, options);

  auto [out_tensor] = get_implementation()->forward(inputs);

  chi = out_tensor[0].item<double>() * MY_PI;
  R = out_tensor[1].item<double>();
  r = out_tensor[2].item<double>();

  return {chi, R, r};


}
void NNCollideModel::setup_model(int training){
  if (comm->me == 0){
    std::cout<< "Getting params" << std::endl;
    read_train_params();
    read_params();
  }

  MPI_Bcast(&NN_params,sizeof(NNParams),MPI_BYTE,0,world);
  MPI_Bcast(&train_params,sizeof(TrainParams),MPI_BYTE,0,world);

  training_data.num_features = 10; 
  training_data.num_outputs = 3;

  int num_hidden    = NN_params.width;
  
  set_implementation(
    new NNCollideModelImpl( training_data.num_features, num_hidden)
  );

  get_implementation()->to(torch::kDouble);

  if ((training == OFFLINE) && (comm->me == 0)){
    this->load_weights(NN_params.model_loc); 
  }

  for (auto& param : get_implementation()->named_parameters()) {
    MPI_Bcast( param.value().data_ptr(),
          param.value().numel(),
          MPI_DOUBLE,
          0, world);
  }

  optimizer = std::make_shared<torch::optim::Adam>(
      get_implementation()->parameters(), torch::optim::AdamOptions(train_params.LR)
    );
}


void NNCollideModel::read_params()
{
  const char* fname = "in.nn_params";
  FILE *fp = fopen(fname,"r");
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open DMS parameter file %s",fname);
    error->one(FLERR,str);
  }
  int REQWORDS = 2;
  char **words = new char*[REQWORDS];
  char line[MAXLINE];
  while (fgets(line,MAXLINE,fp)) {
    int pre = strspn(line," \t\n\r");
    if (pre == strlen(line) || line[pre] == '#') continue;

    int nwords = wordparse(REQWORDS,line,words);
    if (nwords < REQWORDS) 
      error->one(FLERR,"Incorrect line format in DMS parameter file");

    NN_params.width = atoi(words[0]);

    NN_params.model_loc = words[1];

  }
  delete [] words;
  fclose(fp);
}


void NNCollideModel::update_LR(int total_epochs){
  double decayed_LR = train_params.LR * train_params.A / ( train_params.B + train_params.C * total_epochs );

  for (auto &group : (*optimizer).param_groups())
        {
          if(group.has_options())
          {
            auto &options = static_cast<torch::optim::OptimizerOptions &>(group.options());
            options.set_lr( decayed_LR);
          }
        }
}

torch::Tensor NNCollideModel::get_loss(torch::Tensor local_inputs, torch::Tensor train_out){

  auto [outputs] = get_implementation()->forward(local_inputs);
  torch::Tensor loss = (outputs - train_out).square()  ;

  return loss;
}

void NNCollideModel::optimizer_step(){
  optimizer->step();
  optimizer->zero_grad(false);
}

void NNCollideModel::broadcast_weights(){
  for (auto& param : get_implementation()->named_parameters()) {
          MPI_Bcast( param.value().data_ptr(),
                param.value().numel(),
                MPI_DOUBLE,
                0, world);
        }
}

void NNCollideModel::save_weights(int step){
  torch::serialize::OutputArchive output_model_archive;
  get_implementation()->save( output_model_archive);
  output_model_archive.save_to("NN_trained_" + std::to_string(step)+ ".pt");
}