#include "collision_model_vss.h"
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

// Idea is that in the worst case online optimisation of VSS/LB parameters can be a paper if NN approximation of CTC does not work.

VSSCollideModel::VSSCollideModel(SPARTA *sparta) :
  MLCollideModel(sparta)
{
}

VSSCollideModel::~VSSCollideModel(){}



std::tuple<double, double, double> VSSCollideModel::collide(double input_data[]){

  double chi, R, r;

  double e_star = input_data[0];
  double b_star = input_data[1];

  double b = b_star * sigma_LJ * train_params.b_ref
  double dist = sqrt( e_star * ( train_params.e_ref * epsilon_LJ ) *2 / params[isp][jsp].mr);
  double d = get_implementation()->d_ref * np.sqrt(get_implementation()->g_ref_power_2nu()) * ( 1/dist) **(get_implementation()->omega_param - 0.5);
  chi = 2 * arccos( b / d );

  E_dispose = precoln.etrans;
  E_Dispose += p->erot;
  Fraction_Rot =
  1- pow(random->uniform(),
          (1/(2.5-params[ip->ispecies][jp->ispecies].omega)));
  p->erot = Fraction_Rot * E_Dispose;
  E_Dispose -= p->erot;

  E_dispose = precoln.etrans;
  E_Dispose += p->erot;
  Fraction_Rot =
  1- pow(random->uniform(),
          (1/(2.5-params[ip->ispecies][jp->ispecies].omega)));
  p->erot = Fraction_Rot * E_Dispose;
  E_Dispose -= p->erot;

  

}
void VSSCollideModel::setup_model(int training){
  if (comm->me == 0){
    std::cout<< "Getting params" << std::endl;
    read_train_params();
    read_params();
  }

  MPI_Bcast(&VSS_params,sizeof(VSSParams),MPI_BYTE,0,world);
  MPI_Bcast(&train_params,sizeof(TrainParams),MPI_BYTE,0,world);


  optimizer = std::make_shared<torch::optim::Adam>(
      get_implementation()->parameters(), torch::optim::AdamOptions(train_params.LR)
    );
}


void VSSCollideModel::read_params()
{
  const char* fname = "in.VSS_params";
  FILE *fp = fopen(fname,"r");
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open DMS parameter file %s",fname);
    error->one(FLERR,str);
  }
  int REQWORDS = 3;
  char **words = new char*[REQWORDS];
  char line[MAXLINE];
  while (fgets(line,MAXLINE,fp)) {
    int pre = strspn(line," \t\n\r");
    if (pre == strlen(line) || line[pre] == '#') continue;

    int nwords = wordparse(REQWORDS,line,words);
    if (nwords < REQWORDS) 
      error->one(FLERR,"Incorrect line format in DMS parameter file");

    VSS_params.width = atoi(words[0]);
    VSS_params.gaussians = atoi(words[1]);

    VSS_params.model_loc = words[2];

  }
  delete [] words;
  fclose(fp);
}


void VSSCollideModel::update_LR(int total_epochs){
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

torch::Tensor VSSCollideModel::gaussian_distribution( torch::Tensor y, torch::Tensor mu, torch::Tensor sigma){
    torch::Tensor result = (y.index({torch::indexing::Slice(),torch::indexing::None}) - mu) * torch::reciprocal(sigma);
    result = -0.5 * (result * result);
    return(torch::exp(result) * torch::reciprocal(sigma)) * oneDivSqrtTwoPI;
}

torch::Tensor VSSCollideModel::neg_log_likelihood( torch::Tensor pi, torch::Tensor sigma, torch::Tensor mu, torch::Tensor y ){
    torch::Tensor y_hat = torch::log(y/(1-y));
    torch::Tensor result = gaussian_distribution(y_hat, mu, sigma) * pi;
    result = torch::mean(-torch::log(result.sum(1)));
    return( result );
}

torch::Tensor VSSCollideModel::get_loss(torch::Tensor local_inputs, torch::Tensor train_out){

  auto [pi_weights_chi, sigma_chi, mu_chi] = get_implementation()->gen_params_chi(local_inputs);
  torch::Tensor loss_chi = neg_log_likelihood(pi_weights_chi,  sigma_chi, mu_chi, train_out.index({Slice(),0})) ;


  torch::Tensor correlated_inputs = torch::cat({train_out.index({Slice(),0}).index({Slice(),None}),local_inputs},1);
  auto [pi_weights_R, sigma_R, mu_R] = get_implementation()->gen_params_R(correlated_inputs);
  torch::Tensor loss_R = neg_log_likelihood(pi_weights_R,  sigma_R, mu_R, train_out.index({Slice(),1})) ;

  auto [pi_weights_r, sigma_r, mu_r] = get_implementation()->gen_params_r(correlated_inputs);
  torch::Tensor loss_r = neg_log_likelihood(pi_weights_r, sigma_r, mu_r, train_out.index({Slice(),2})) ;

  return loss_chi + loss_R + loss_r;
}

void VSSCollideModel::optimizer_step(){
  optimizer->step();
  optimizer->zero_grad(false);
}

void VSSCollideModel::broadcast_weights(){
  for (auto& param : get_implementation()->named_parameters()) {
          MPI_Bcast( param.value().data_ptr(),
                param.value().numel(),
                MPI_DOUBLE,
                0, world);
        }
}

void VSSCollideModel::save_weights(int step){
  torch::serialize::OutputArchive output_model_archive;
  get_implementation()->save( output_model_archive);
  output_model_archive.save_to("VSS_trained_" + std::to_string(step)+ ".pt");
}