#include "collision_model_mdn.h"
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

MDNCollideModel::MDNCollideModel(SPARTA *sparta) :
  MLCollideModel(sparta)
{
}

MDNCollideModel::~MDNCollideModel(){}

torch::Tensor MDNCollideModel::draw_gumbel(double loc, double scale, torch::Tensor example){
    torch::Tensor p = torch::rand_like(example);
    torch::Tensor z = loc * torch::ones_like(example) - scale * torch::log(-torch::log(p));
    return(z);
}

torch::Tensor MDNCollideModel::forward(torch::Tensor pi_weights, torch::Tensor sigma, torch::Tensor mu){
    torch::Tensor gumbel = draw_gumbel(0,1, pi_weights);
    torch::Tensor k = torch::argmax( torch::log(pi_weights) + gumbel, -1);
    torch::Tensor rn = torch::randn(1);
    torch::Tensor out = rn * sigma.index({k} ) + mu.index({ k});
    return(torch::sigmoid(out));
}

std::tuple<double, double, double> MDNCollideModel::collide(double input_data[]){

  double chi, R, r;

  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor inputs = torch::from_blob(input_data, {6}, options);

  auto [chi_pi, chi_sigma, chi_mu] = get_implementation()->gen_params_chi(inputs);
  torch::Tensor chi_tensor = forward(chi_pi, chi_sigma, chi_mu);


  chi = chi_tensor[0].item<double>() * MY_PI;

  torch::Tensor correlation_inputs = torch::cat({chi_tensor,inputs});
  
  auto [R_pi, R_sigma, R_mu]= get_implementation()->gen_params_R(correlation_inputs);
  R = forward(R_pi, R_sigma, R_mu)[0].item<double>();
  auto [r_pi, r_sigma, r_mu] = get_implementation()->gen_params_r(correlation_inputs);
  r = forward(r_pi, r_sigma, r_mu)[0].item<double>();

  return {chi, R, r};


}
void MDNCollideModel::setup_model(int training){
  if (comm->me == 0){
    std::cout<< "Getting params" << std::endl;
    read_train_params();
    read_params();
  }

  MPI_Bcast(&mdn_params,sizeof(MDNParams),MPI_BYTE,0,world);
  MPI_Bcast(&train_params,sizeof(TrainParams),MPI_BYTE,0,world);

  training_data.num_features = 6; 
  training_data.num_outputs = 3;

  int num_gaussians = mdn_params.gaussians;
  int num_hidden    = mdn_params.width;
  
  set_implementation(
    new MDNCollideModelImpl( training_data.num_features, num_hidden, num_gaussians)
  );

  if (training == OFFLINE) {
    this->load_weights(mdn_params.chi_model); 
  }

  get_implementation()->to(torch::kDouble);

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


void MDNCollideModel::read_params()
{
  const char* fname = "in.mdn_params";
  FILE *fp = fopen(fname,"r");
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open DMS parameter file %s",fname);
    error->one(FLERR,str);
  }
  int REQWORDS = 5;
  char **words = new char*[REQWORDS];
  char line[MAXLINE];
  while (fgets(line,MAXLINE,fp)) {
    int pre = strspn(line," \t\n\r");
    if (pre == strlen(line) || line[pre] == '#') continue;

    int nwords = wordparse(REQWORDS,line,words);
    if (nwords < REQWORDS) 
      error->one(FLERR,"Incorrect line format in DMS parameter file");

    mdn_params.width = atoi(words[0]);
    mdn_params.gaussians = atoi(words[1]);

    mdn_params.chi_model = words[2];
    mdn_params.R_model = words[3];
    mdn_params.r_model = words[4];

  }
  delete [] words;
  fclose(fp);
}


void MDNCollideModel::update_LR(int total_epochs){
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

torch::Tensor MDNCollideModel::gaussian_distribution( torch::Tensor y, torch::Tensor mu, torch::Tensor sigma){
    torch::Tensor result = (y.index({torch::indexing::Slice(),torch::indexing::None}) - mu) * torch::reciprocal(sigma);
    result = -0.5 * (result * result);
    return(torch::exp(result) * torch::reciprocal(sigma)) * oneDivSqrtTwoPI;
}

torch::Tensor MDNCollideModel::neg_log_likelihood( torch::Tensor pi, torch::Tensor sigma, torch::Tensor mu, torch::Tensor y ){
    torch::Tensor y_hat = torch::log(y/(1-y));
    torch::Tensor result = gaussian_distribution(y_hat, mu, sigma) * pi;
    result = torch::mean(-torch::log(result.sum(1)));
    return( result );
}

torch::Tensor MDNCollideModel::get_loss(torch::Tensor local_inputs, torch::Tensor train_out){

  auto [pi_weights_chi, sigma_chi, mu_chi] = get_implementation()->gen_params_chi(local_inputs);
  torch::Tensor loss_chi = neg_log_likelihood(pi_weights_chi,  sigma_chi, mu_chi, train_out.index({Slice(),0})) ;


  torch::Tensor correlated_inputs = torch::cat({train_out.index({Slice(),0}).index({Slice(),None}),local_inputs},1);
  auto [pi_weights_R, sigma_R, mu_R] = get_implementation()->gen_params_R(correlated_inputs);
  torch::Tensor loss_R = neg_log_likelihood(pi_weights_R,  sigma_R, mu_R, train_out.index({Slice(),1})) ;

  auto [pi_weights_r, sigma_r, mu_r] = get_implementation()->gen_params_r(correlated_inputs);
  torch::Tensor loss_r = neg_log_likelihood(pi_weights_r, sigma_r, mu_r, train_out.index({Slice(),2})) ;

  return loss_chi + loss_R + loss_r;
}

void MDNCollideModel::optimizer_step(){
  optimizer->step();
  optimizer->zero_grad(false);
}

void MDNCollideModel::broadcast_weights(){
  for (auto& param : get_implementation()->named_parameters()) {
          MPI_Bcast( param.value().data_ptr(),
                param.value().numel(),
                MPI_DOUBLE,
                0, world);
        }
}

void MDNCollideModel::save_weights(int step){
  torch::serialize::OutputArchive output_model_archive;
  get_implementation()->save( output_model_archive);
  output_model_archive.save_to("mdn_trained.pt");
}