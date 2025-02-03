#include "collide_dms.h"
#include "math_const.h"
#include "comm.h"
#include "math.h"
#include "random_knuth.h"
#include "mixture.h"
#include "string.h"
#include <torch/torch.h>
#include "collide_nn.h"

#include "stdlib.h"
#include "error.h"
#include "update.h"

using namespace SPARTA_NS;
using namespace MathConst;
using namespace torch::indexing;

#define MAXLINE 1024

enum{NO,START,ALL, OFFLINE};

CollideDMS::CollideDMS(SPARTA *sparta, int narg, char **arg) :
  Collide(sparta,narg,arg)
{ 
  training = NO; // TODO: Parsing logic
  //printf("Training %.d\n", training);
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"train") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal collide command");
      if (strcmp(arg[iarg+1],"none") == 0) training = NO;
      else if (strcmp(arg[iarg+1],"start") == 0) training = START;
      else if (strcmp(arg[iarg+1],"all") == 0) training = ALL;
      else if (strcmp(arg[iarg+1],"offline") == 0) training = OFFLINE;
      else error->all(FLERR,"Illegal collide command");
      iarg += 2;
    } else error->all(FLERR,"Illegal collide command");
  }

  //printf("Training %.d\n", training);

  nparams = particle->nspecies;
  if (nparams == 0)
    error->all(FLERR,"Cannot use collide command with no species defined");

  memory->create(params,nparams,nparams,"collide:params");
  if (comm->me == 0) read_param_file(arg[2]);
  MPI_Bcast(params[0],nparams*nparams*sizeof(Params),MPI_BYTE,0,world);
  if (training) setup_model();
}
CollideDMS::~CollideDMS()
{
  if (copymode) return;

  memory->destroy(params);
}

void CollideDMS::init()
{
  // initially read-in per-species params must match current species list

  if (nparams != particle->nspecies)
    error->all(FLERR,"DMS parameters do not match current species");

  Collide::init();
}

void CollideDMS::setup_model(){
  // Temporary test

  int num_features = 10;
  int width = 50;
  int num_outputs = 3;

  CollisionModel = std::make_shared<NNModel>(
    NNModel(num_features, width, num_outputs)
  );

  if (training == OFFLINE) {
    // CollisionModel.load_parameters("model.pt"); // TODO: Read from params file?
    torch::serialize::InputArchive input_archive;
    input_archive.load_from("model_trained.pt");
    (*CollisionModel).load(input_archive);
  }

  (*CollisionModel).to(torch::kDouble);
   
  for (auto& param : (*CollisionModel).named_parameters()) {
    MPI_Bcast( param.value().data_ptr(),
          param.value().numel(),
          MPI_DOUBLE,
          0, world);
  }

  // Test parameter initialisation. TODO: To be replaced by parsing logic.
  if (training == START) {
    train_params.train_every = 1;
    train_params.train_max = 20;
    train_params.epochs=100;
    train_params.len_data=64000/comm->nprocs; // Some processes will not have this many collisions!
    train_params.LR=1e-3;
    train_params.A = 400.; 
    train_params.B = 400.; 
    train_params.C = 1.; 
    train_params.batch_size = 250;
  } else if (training == ALL) {
    train_params.train_every = 20;
    train_params.train_max = 1000; // TODO: this should not be hard coded
    train_params.epochs=100;
    train_params.len_data=54000;
    train_params.LR=1e-3;
    train_params.A = 400.; 
    train_params.B = 400.; 
    train_params.C = 1.; 
    train_params.batch_size = 250;
  }

  optimizer = std::make_shared<torch::optim::RMSprop>(
    (*CollisionModel).parameters(), torch::optim::RMSpropOptions(train_params.LR)
  );

  total_epochs = 0;

  training_data.num_features = num_features;
  training_data.num_outputs = num_outputs;
}

int CollideDMS::train_this_step(int step){
  if (training == NO || training == OFFLINE) return 0;
  return (step % train_params.train_every == 0) && (step < train_params.train_max);
}

void CollideDMS::train(int step){
  if (!train_this_step(step)){
    return;
  } else {
    int N_data = MIN( train_params.len_data, training_data.outputs.size() / training_data.num_outputs);
    std::cout << "Data: " << N_data << " Process: " << comm->me<<std::endl; 
    auto options = torch::TensorOptions().dtype(torch::kFloat64);

    torch::Tensor inputs = torch::from_blob(training_data.features.data(), {N_data, training_data.num_features}, options);
    torch::Tensor chi = torch::from_blob(training_data.outputs.data(), {N_data, training_data.num_outputs}, options);

    for(int l=0;l<train_params.epochs;l++){

      torch::Tensor shuffled_indices = torch::randperm(N_data, torch::TensorOptions().dtype(at::kLong));

      chi = chi.index({shuffled_indices});
      inputs = inputs.index({shuffled_indices});

      // Decay learning rate

      for (auto &group : (*optimizer).param_groups())
      {
        if(group.has_options())
        {
          auto &options = static_cast<torch::optim::OptimizerOptions &>(group.options());
          options.set_lr(train_params.LR * train_params.A / ( train_params.B + train_params.C * total_epochs ) );
        }
      }

      for (int p=0; p<train_params.len_data; p=p+train_params.batch_size) {

        if ( p+train_params.batch_size<N_data) {
          Slice slice(p, p+train_params.batch_size);
          torch::Tensor pred = (*CollisionModel).forward(inputs.index({slice}));
          torch::Tensor loss = (pred - chi.index({slice})).square().sum(); // Change to mean and also adjust LR.

          loss.backward();

          double total_loss;

          MPI_Allreduce(&total_loss, loss.data_ptr(),
                loss.numel(),
                  MPI_DOUBLE,
                  MPI_SUM, world); // loss has not been updated on some ranks.
        } else {
          // We have run out data on this process.
          (*optimizer).zero_grad(false);

	      }
        // MPI reduce the gradients
        for (auto& param : (*CollisionModel).named_parameters()) {
          MPI_Barrier(world);
          MPI_Allreduce(MPI_IN_PLACE, param.value().grad().data_ptr(),
               param.value().grad().numel(),
                MPI_DOUBLE,
                MPI_SUM, world);
          MPI_Barrier(world);
          //param.value().grad().data() = param.value().grad().data() / comm->nprocs;
        }

        (*optimizer).step();
        (*optimizer).zero_grad(false);
        

        // printstring = f"{i}, {j}, {train_loss}, {test_loss}, {g['lr']}, {N_train}, {batch_size}"
      }
      total_epochs++;
    }
  // Delete training data so that it can be rebuilt for next training step.
  training_data.features.clear();
  training_data.outputs.clear();
  
  if (comm->me == 0) {
    torch::serialize::OutputArchive output_model_archive;
    (*CollisionModel).save( output_model_archive);
    output_model_archive.save_to("model_trained.pt");
  }
  }
}

double CollideDMS::vremax_init(int igroup, int jgroup)
{
  Particle::Species *species = particle->species;
  double *vscale = mixture->vscale; // For example initial thermal velocity of mixture
  int *mix2group = mixture->mix2group;
  int nspecies = particle->nspecies;

  double vrmgroup = 0.0;

  for (int isp = 0; isp < nspecies; isp++) {
    if (mix2group[isp] != igroup) continue;
    for (int jsp = 0; jsp < nspecies; jsp++) {
      if (mix2group[jsp] != jgroup) continue;

      double cxs = 12*params[isp][jsp].sigma*params[isp][jsp].sigma*MY_PI; // This is not a good estimate
      double beta = MAX(vscale[isp],vscale[jsp]);
      double vrm = 2.0 * cxs * beta;

      vrmgroup = MAX(vrmgroup,vrm);
    }
  }

  return vrmgroup;
}

double CollideDMS::attempt_collision(int icell, int np, double volume)
{
  double fnum = update->fnum;
  double dt = update->dt;

  double nattempt;

  if (remainflag) {
    nattempt = 0.5 * np * (np-1) *
      vremax[icell][0][0] * dt * fnum / volume + remain[icell][0][0];
    remain[icell][0][0] = nattempt - static_cast<int> (nattempt);
  } else {
    nattempt = 0.5 * np * (np-1) *
      vremax[icell][0][0] * dt * fnum / volume + random->uniform();
  }

  return nattempt;
}


double CollideDMS::attempt_collision(int icell, int igroup, int jgroup,
                                     double volume)
{
 double fnum = update->fnum;
 double dt = update->dt;

 double nattempt;

 // return 2x the value for igroup != jgroup, since no J,I pairing

 double npairs;
 if (igroup == jgroup) npairs = 0.5 * ngroup[igroup] * (ngroup[igroup]-1);
 else npairs = ngroup[igroup] * (ngroup[jgroup]);
 //else npairs = 0.5 * ngroup[igroup] * (ngroup[jgroup]);

 nattempt = npairs * vremax[icell][igroup][jgroup] * dt * fnum / volume;

 if (remainflag) {
   nattempt += remain[icell][igroup][jgroup];
   remain[icell][igroup][jgroup] = nattempt - static_cast<int> (nattempt);
 } else nattempt += random->uniform();

 return nattempt;
}

int CollideDMS::test_collision(int icell, int igroup, int jgroup,
                               Particle::OnePart *ip, Particle::OnePart *jp)
{
  double *vi = ip->v;
  double *vj = jp->v;
  int ispecies = ip->ispecies;
  int jspecies = jp->ispecies;
  double du  = vi[0] - vj[0];
  double dv  = vi[1] - vj[1];
  double dw  = vi[2] - vj[2];
  double vr2 = du*du + dv*dv + dw*dw;
  double vro  = pow(vr2,0.5);

  // although the vremax is calculated for the group,
  // the individual collisions calculated species dependent vre

  double b = (params[ispecies][jspecies].A * 
    pow( vro, params[ispecies][jspecies].B ) + params[ispecies][jspecies].C) 
    * params[ispecies][jspecies].sigma;
  precoln.bmax = b;
  double vre = vro*b*b*MY_PI;
  vremax[icell][igroup][jgroup] = MAX(vre,vremax[icell][igroup][jgroup]);
  if (vre/vremax[icell][igroup][jgroup] < random->uniform()) return 0;
  precoln.vr2 = vr2;
  return 1;
}

void CollideDMS::setup_collision(Particle::OnePart *ip, Particle::OnePart *jp)
{
  Particle::Species *species = particle->species;

  int isp = ip->ispecies;
  int jsp = jp->ispecies;

  precoln.vr = sqrt(precoln.vr2);

  precoln.ave_rotdof = 0.5 * (species[isp].rotdof + species[jsp].rotdof);
  precoln.ave_vibdof = 0.5 * (species[isp].vibdof + species[jsp].vibdof);
  precoln.ave_dof = (precoln.ave_rotdof  + precoln.ave_vibdof)/2.;

  double imass = precoln.imass = species[isp].mass;
  double jmass = precoln.jmass = species[jsp].mass;

  precoln.etrans = 0.5 * params[isp][jsp].mr * precoln.vr2;
  precoln.erot = ip->erot + jp->erot;
  precoln.evib = ip->evib + jp->evib;

  precoln.eint   = precoln.erot + precoln.evib;
  precoln.etotal = precoln.etrans + precoln.eint;

  double divisor = 1.0 / (imass+jmass);
  double *vi = ip->v;
  double *vj = jp->v;
  precoln.ucmf = ((imass*vi[0])+(jmass*vj[0])) * divisor;
  precoln.vcmf = ((imass*vi[1])+(jmass*vj[1])) * divisor;
  precoln.wcmf = ((imass*vi[2])+(jmass*vj[2])) * divisor;

  precoln.D_cutoff = MAX( 4*params[isp][jsp].sigma, 1.5*precoln.bmax);
}

int CollideDMS::perform_collision(Particle::OnePart *&ip,
                                  Particle::OnePart *&jp,
                                  Particle::OnePart *&kp)
{
  int reactflag;
  if (react) // raise some kind of error?
    error->one(FLERR,"Reaction chemistry not implemented for DMS collision");
  else
    reactflag=0;
    if (precoln.ave_vibdof > 0.0 ) {
      error->all(FLERR,"Scattering not implemented for vibrating molecules.");
      SCATTER_VibDiatomicScatter(ip,jp);
    } else if (precoln.ave_rotdof >0.0 ) {
      SCATTER_RigidDiatomicScatter(ip,jp);
    } else {
      SCATTER_MonatomicScatter(ip,jp);
    }


  return reactflag;
}

void CollideDMS::SCATTER_MonatomicScatter(
  Particle::OnePart *ip,
  Particle::OnePart *jp
)
{   
  Particle::Species *species = particle->species;
  int isp = ip->ispecies;
  int jsp = jp->ispecies;
  double mass_i = species[isp].mass;
  double mass_j = species[jsp].mass;
  // Generate b

  double ua,vb,wc;
  double vrc[3];

  double dt = params[isp][jsp].dt_verlet;
  double sigma_LJ = params[isp][jsp].sigma;
  double epsilon_LJ = params[isp][jsp].epsilon;
  
  double b = pow(random->uniform(), 0.5) * precoln.bmax; 

  double coschi, sinchi;

  int trajectory = (!training)||(training_data.outputs.size() < train_params.len_data ) ;
  if (trajectory)
  {
    double dist, d;
    double f1[2], f2, x2s[2], x1s[2], v1s[2], v2s[2];
    
    x1s[0]=0.;
    x1s[1]=0.;

    x2s[0] = precoln.D_cutoff;
    x2s[1] = b;
    
    v1s[0] = precoln.vr;
    v1s[1] = 0.;

    v2s[0]=0.;
    v2s[1]=0.;

    double v_cm_x = (v1s[0] * mass_i+ v2s[0] *mass_j) / (mass_i+mass_j);
    double v_cm_y = (v1s[1]*mass_i+ v2s[1]*mass_j ) / (mass_i+mass_j); 
    v1s[0] = v1s[0] - v_cm_x;
    v2s[0] = v2s[0] - v_cm_x;
    v1s[1] = v1s[1] - v_cm_y;
    v2s[1] = v2s[1] - v_cm_y;

    for (int i=0;i<params[isp][jsp].timesteps;i++){
      dist = sqrt(  pow( x1s[0]-x2s[0], 2) + pow(x1s[1]-x2s[1], 2) );
      d = sigma_LJ / dist;
      for (int k=0;k<2;k++){
        f1[k]  = (( x1s[k] - x2s[k] ) / dist )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(d,13) - pow(d,7)) ;
        x1s[k] = x1s[k] + v1s[k] * dt - 0.5 * ( f1[k]/mass_i ) * pow(dt,2);
        x2s[k] = x2s[k] + v2s[k] * dt + 0.5 * ( f1[k]/mass_j ) * pow(dt,2);
      }

      dist = sqrt(  pow( x1s[0]-x2s[0], 2) + pow(x1s[1]-x2s[1], 2) );

      d = sigma_LJ / dist;
      for (int k=0;k<2;k++){
        f2 = (( x1s[k] - x2s[k] ) / dist )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(d,13) - pow(d,7));
        v1s[k] = v1s[k] - 0.5 * ( f1[k] + f2 ) / mass_i * dt;
        v2s[k] = v2s[k] + 0.5 * ( f1[k] + f2 ) / mass_j * dt;
      }

      if ( i>200 && dist>precoln.D_cutoff ){
        break;
      }
    }
    coschi = v1s[0] / sqrt( pow(v1s[0],2) +  pow(v1s[1],2) );
    
    if (training && training_data.outputs.size() /training_data.num_outputs < train_params.len_data  ){
      double e_ref = 100;
      double b_ref = 5;
      double e_star = precoln.etrans / (epsilon_LJ * e_ref);
      double b_star = b / (sigma_LJ * b_ref);
      training_data.features.push_back(e_star);
      training_data.features.push_back(b_star);
      training_data.outputs.push_back(acos(coschi)/MY_PI );
    }
  }
  else
  { 
    double e_ref = 100;
    double b_ref = 5;
    double e_star = precoln.etrans / (epsilon_LJ * e_ref);
    double b_star = b / (sigma_LJ * b_ref);
    double input_data[] = {e_star, b_star};
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor inputs = torch::from_blob(input_data, {training_data.num_features}, options);
    torch::Tensor pred = (*CollisionModel).forward(inputs);
    double chi = *pred.data_ptr<double>() * MY_PI;
    coschi = cos( chi );
  }

  sinchi = sqrt( 1 - coschi*coschi );
  double eps = random->uniform() * MY_2PI;
  double *vi = ip->v;
  double *vj = jp->v;

  vrc[0] = vi[0]-vj[0];
  vrc[1] = vi[1]-vj[1];
  vrc[2] = vi[2]-vj[2];

  postcoln.etrans = precoln.etrans;
  double scale = sqrt((2.0 * postcoln.etrans) / (params[isp][jsp].mr * precoln.vr2));
  double d = sqrt(vrc[1]*vrc[1]+vrc[2]*vrc[2]);
  if (d > 1.0e-6) {
    ua = scale * ( coschi*vrc[0] + sinchi*d*sin(eps) );
    vb = scale * ( coschi*vrc[1] + sinchi*(precoln.vr*vrc[2]*cos(eps) -
                                        vrc[0]*vrc[1]*sin(eps))/d );
    wc = scale * ( coschi*vrc[2] - sinchi*(precoln.vr*vrc[1]*cos(eps) +
                                        vrc[0]*vrc[2]*sin(eps))/d );
  } else {
    ua = scale * ( coschi*vrc[0] );
    vb = scale * ( sinchi*vrc[0]*cos(eps) );
    wc = scale * ( sinchi*vrc[0]*sin(eps) );
  }

  double divisor = 1.0 / (mass_i + mass_j);
  vi[0] = precoln.ucmf + (mass_j*divisor)*ua;
  vi[1] = precoln.vcmf + (mass_j*divisor)*vb;
  vi[2] = precoln.wcmf + (mass_j*divisor)*wc;
  vj[0] = precoln.ucmf - (mass_i*divisor)*ua;
  vj[1] = precoln.vcmf - (mass_i*divisor)*vb;
  vj[2] = precoln.wcmf - (mass_i*divisor)*wc;
}

void CollideDMS::SCATTER_VibDiatomicScatter(
  Particle::OnePart *ip,
  Particle::OnePart *jp
)
{}

void CollideDMS::SCATTER_RigidDiatomicScatter(
  Particle::OnePart *ip,
  Particle::OnePart *jp
)
{  
  // If we have arrived here assume two diatomic molecules. 
  Particle::Species *species = particle->species;
  int isp = ip->ispecies;
  int jsp = jp->ispecies;
  double mass_i = species[isp].mass;
  double mass_j = species[jsp].mass;

  double bond_length_i = params[isp][jsp].bond_length_i; // TODO: Add some logic to set this.
  double bond_length_j = params[isp][jsp].bond_length_j;

  // The two atomic masses within a molecule must at the moment be the same i.e. N2, O2.
  double atom_mass_i = mass_i/2;
  double atom_mass_j = mass_j/2;

  double I1 = atom_mass_i/2 * pow( bond_length_i, 2);
  double I2 = atom_mass_j/2 * pow( bond_length_j, 2);

  double ua,vb,wc;
  double vrc[3];

  double dt = params[isp][jsp].dt_verlet;
  double sigma_LJ = params[isp][jsp].sigma;
  double epsilon_LJ = params[isp][jsp].epsilon;
  double f11_12[3], f11_21[3], f11_22[3], f12_21[3], f12_22[3], f21_22[3];
  double f11[3], f12[3], f21[3], f22[3];
  double q11[3], q12[3], q21[3], q22[3];
  double x11s[3], x12s[3], x21s[3], x22s[3], v11s[3], v12s[3], v21s[3], v22s[3];

  double dt_dsmc = update->dt;

  double g1, g2, s1[3], s2[3];
  double d;
  double tol = 1e-16;

  double d_11_21, d_11_12 ;
  double d_11_22;
  double d_12_21;
  double d_12_22, d_21_22;

  double err1,err2, k1, k2;

  double coschi, erot1_new, erot2_new;

  // Setup the initial conditions
  double x0_1, x0_2,y0_1, y0_2;

  double b = pow(random->uniform(), 0.5) * precoln.bmax;
  double theta1 = acos( 2.0*random->uniform() - 1.0);
  double theta2 = acos( 2.0*random->uniform() - 1.0);

  double phi1 = random->uniform()*MY_2PI;
  double phi2 = random->uniform()*MY_2PI;

  double eta1 = random->uniform()*MY_2PI;
  double eta2 = random->uniform()*MY_2PI;

  int trajectory = (!training)||(training_data.outputs.size() / training_data.num_outputs< train_params.len_data ) ;
  if (trajectory)
  {

    // Particle j initially stationary at (D_cutoff, b)
    x21s[0] = x22s[0] = precoln.D_cutoff;
    x21s[1] = x22s[1] = b;
    x21s[2] = x22s[2] = 0.;

    v21s[0] = v22s[0] = 0.;
    v21s[1] = v22s[1] = 0.;
    v21s[2] = v22s[2] = 0.;

    // Particle i initially at the origin, with velocity (vr,0,0)
    x11s[0] = x12s[0] = 0.;
    x11s[1] = x12s[1] = 0.;
    x11s[2] = x12s[2] = 0.;

    v11s[0] = v12s[0] = precoln.vr;
    v11s[1] = v12s[1] = 0.;
    v11s[2] = v12s[2] = 0.;

    // Atoms displaced from centre of mass
    
    x11s[0] += cos( phi1 ) * sin( theta1 ) * bond_length_i / 2.;
    x12s[0] -= cos( phi1 ) * sin( theta1 ) * bond_length_i / 2.;
    x11s[1] += sin( phi1 ) * sin( theta1 )* bond_length_i / 2.;
    x12s[1] -= sin( phi1 ) * sin( theta1 )* bond_length_i / 2.;
    x11s[2] += cos(theta1)* bond_length_i / 2.;
    x12s[2] -= cos(theta1)* bond_length_i / 2.;

    x21s[0] += cos( phi2 ) * sin( theta2 ) * bond_length_j / 2.;
    x22s[0] -= cos( phi2 ) * sin( theta2 ) * bond_length_j / 2.;
    x21s[1] += sin( phi2 ) * sin( theta2 )* bond_length_j / 2.;
    x22s[1] -= sin( phi2 ) * sin( theta2 )* bond_length_j / 2.;
    x21s[2] += cos(theta2)* bond_length_j / 2.;
    x22s[2] -= cos(theta2)* bond_length_j / 2.;

    // Molecules have angular velocity in random plane. Choose particular velocities perpendicular to the displacement vector.

    v11s[0] += (cos( phi1 ) * cos( theta1 )* cos(eta1) - sin(phi1)*sin(eta1)) * sqrt( 2 * ip->erot / I1 ) * bond_length_i /2;
    v12s[0] -= (cos( phi1 ) * cos( theta1 )* cos(eta1) - sin(phi1)*sin(eta1)) * sqrt( 2 * ip->erot / I1 ) * bond_length_i /2;
    v11s[1] += (sin( phi1 ) * cos( theta1 )* cos(eta1) + cos(phi1)*sin(eta1)) * sqrt( 2 * ip->erot / I1 ) * bond_length_i /2;
    v12s[1] -= (sin( phi1 ) * cos( theta1 )* cos(eta1) + cos(phi1)*sin(eta1)) * sqrt( 2 * ip->erot / I1 ) * bond_length_i /2;
    v11s[2] += -sin(theta1)* cos(eta1)*sqrt( 2 * ip->erot / I1 ) * bond_length_i/2 ;
    v12s[2] -= -sin(theta1)* cos(eta1)*sqrt( 2 * ip->erot / I1 ) * bond_length_i /2;

    v21s[0] += (cos( phi2 ) * cos( theta2 )* cos(eta2) - sin(phi2)*sin(eta2)) * sqrt( 2 * jp->erot / I2 ) * bond_length_j/2 ;
    v22s[0] -= (cos( phi2 ) * cos( theta2 )* cos(eta2) - sin(phi2)*sin(eta2)) * sqrt( 2 * jp->erot / I2 ) * bond_length_j/2 ;
    v21s[1] += (sin( phi2 ) * cos( theta2 )* cos(eta2) + cos(phi2)*sin(eta2)) * sqrt( 2 * jp->erot / I2 ) * bond_length_j/2 ;
    v22s[1] -= (sin( phi2 ) * cos( theta2 )* cos(eta2) + cos(phi2)*sin(eta2)) * sqrt( 2 * jp->erot / I2 ) * bond_length_j /2;
    v21s[2] += -sin(theta2)* cos(eta2)*sqrt( 2 * jp->erot / I2 ) * bond_length_j /2;
    v22s[2] -= -sin(theta2)* cos(eta2)*sqrt( 2 * jp->erot / I2 ) * bond_length_j /2;

    double vcm;
    for (int k=0;k<3;k++){
      vcm = (atom_mass_i * ( v11s[k] +v12s[k] ) + atom_mass_j * (v21s[k] +v22s[k]))/ (2*atom_mass_i + 2*atom_mass_j);
      v11s[k] -= vcm;
      v12s[k] -= vcm;
      v21s[k] -= vcm;
      v22s[k] -= vcm;
    // Transform to centre of mass frame
    }

    double d_initial = sqrt(  pow( x11s[0]-x22s[0], 2) + pow(x11s[1]-x22s[1], 2) + pow(x11s[2]-x22s[2], 2) );

    for (int i=0;i<params[isp][jsp].timesteps;i++){
      d_11_12 = sqrt(  pow( x11s[0]-x12s[0], 2) + pow(x11s[1]-x12s[1], 2) + pow(x11s[2]-x12s[2], 2) );
      d_11_21 = sqrt(  pow( x11s[0]-x21s[0], 2) + pow(x11s[1]-x21s[1], 2) + pow(x11s[2]-x21s[2], 2) );
      d_11_22 = sqrt(  pow( x11s[0]-x22s[0], 2) + pow(x11s[1]-x22s[1], 2) + pow(x11s[2]-x22s[2], 2) );
      d_12_21 = sqrt(  pow( x12s[0]-x21s[0], 2) + pow(x12s[1]-x21s[1], 2) + pow(x12s[2]-x21s[2], 2) );
      d_12_22 = sqrt(  pow( x12s[0]-x22s[0], 2) + pow(x12s[1]-x22s[1], 2) + pow(x12s[2]-x22s[2], 2) );
      d_21_22 = sqrt(  pow( x21s[0]-x22s[0], 2) + pow(x21s[1]-x22s[1], 2) + pow(x21s[2]-x22s[2], 2) );

      for (int k=0;k<3;k++){
        f11_12[k]  = (( x11s[k] - x12s[k] ) / d_11_12 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_11_12,13) - pow(sigma_LJ/d_11_12,7)) ;
        f11_21[k]  = (( x11s[k] - x21s[k] ) / d_11_21 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_11_21,13) - pow(sigma_LJ/d_11_21,7)) ;
        f11_22[k]  = (( x11s[k] - x22s[k] ) / d_11_22 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_11_22,13) - pow(sigma_LJ/d_11_22,7)) ;
        f12_21[k]  = (( x12s[k] - x21s[k] ) / d_12_21 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_12_21,13) - pow(sigma_LJ/d_12_21,7)) ;
        f12_22[k]  = (( x12s[k] - x22s[k] ) / d_12_22 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_12_22,13) - pow(sigma_LJ/d_12_22,7)) ;
        f21_22[k]  = (( x21s[k] - x22s[k] ) / d_21_22 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_21_22,13) - pow(sigma_LJ/d_21_22,7)) ;

        f11[k] =  - f11_21[k] - f11_22[k] - f11_12[k];
        f12[k] =  - f12_21[k] - f12_22[k] + f11_12[k];
        f21[k] =  f12_21[k] + f11_21[k] - f21_22[k];
        f22[k] = f12_22[k] + f11_22[k] + f21_22[k];

        q11[k] = v11s[k] + 0.5 * ( f11[k]/atom_mass_i ) * dt ;
        q12[k] = v12s[k] + 0.5 * ( f12[k]/atom_mass_i ) * dt ;
        q21[k] = v21s[k] + 0.5 * ( f21[k]/atom_mass_j ) * dt ;
        q22[k] = v22s[k] + 0.5 * ( f22[k]/atom_mass_j ) * dt ;
      }

      // RATTLE part 1: update q vectors to enforce bond constraint.
      while ( true ) {
        for (int k=0;k<3;k++){
          s1[k] = x11s[k] - x12s[k] + dt * ( q11[k] - q12[k] );
          s2[k] = x21s[k] - x22s[k] + dt * ( q21[k] - q22[k] );
        }

        err1 = abs( sqrt(  pow( s1[0], 2) + pow(s1[1], 2) + pow(s1[2], 2) ) - bond_length_i);
        err2 = abs( sqrt(  pow( s2[0], 2) + pow(s2[1], 2) + pow(s2[2], 2) ) - bond_length_j);

        if (err1 < tol && err2 < tol){
          break;
        }

        g1 = ( pow( s1[0], 2) + pow(s1[1], 2) + pow(s1[2], 2) - pow( bond_length_i, 2) ) / (( 2 * dt ) * ( s1[0]*(x11s[0] - x12s[0]) + s1[1]*(x11s[1] - x12s[1]) + s1[2]*(x11s[2] - x12s[2])) * ( 2/atom_mass_i));
        g2 = ( pow( s2[0], 2) + pow(s2[1], 2) + pow(s2[2], 2) - pow( bond_length_j, 2) ) / (( 2 * dt ) * ( s2[0]*(x21s[0] - x22s[0]) + s2[1]*(x21s[1] - x22s[1]) + s2[2]*(x21s[2] - x22s[2])) * ( 2/atom_mass_j));
        
        for (int k=0;k<3;k++){
          q11[k] -= (g1 * (x11s[k] - x12s[k]) / atom_mass_i);
          q12[k] += (g1 * (x11s[k] - x12s[k]) / atom_mass_i);

          q21[k] -= (g2 * (x21s[k] - x22s[k]) / atom_mass_j);
          q22[k] += (g2 * (x21s[k] - x22s[k]) / atom_mass_j);
        }
      }

      for (int k=0;k<3;k++){
        // Update positions and recalculate forces.
        x11s[k] = x11s[k] + dt * q11[k];
        x12s[k] = x12s[k] + dt * q12[k];
        x21s[k] = x21s[k] + dt * q21[k];
        x22s[k] = x22s[k] + dt * q22[k];
      }

      d_11_12 = sqrt(  pow( x11s[0]-x12s[0], 2) + pow(x11s[1]-x12s[1], 2) + pow(x11s[2]-x12s[2], 2) );
      d_11_21 = sqrt(  pow( x11s[0]-x21s[0], 2) + pow(x11s[1]-x21s[1], 2) + pow(x11s[2]-x21s[2], 2) );
      d_11_22 = sqrt(  pow( x11s[0]-x22s[0], 2) + pow(x11s[1]-x22s[1], 2) + pow(x11s[2]-x22s[2], 2) );
      d_12_21 = sqrt(  pow( x12s[0]-x21s[0], 2) + pow(x12s[1]-x21s[1], 2) + pow(x12s[2]-x21s[2], 2) );
      d_12_22 = sqrt(  pow( x12s[0]-x22s[0], 2) + pow(x12s[1]-x22s[1], 2) + pow(x12s[2]-x22s[2], 2) );
      d_21_22 = sqrt(  pow( x21s[0]-x22s[0], 2) + pow(x21s[1]-x22s[1], 2) + pow(x21s[2]-x22s[2], 2) );

      for (int k=0;k<3;k++){
        f11_12[k]  = (( x11s[k] - x12s[k] ) / d_11_12 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_11_12,13) - pow(sigma_LJ/d_11_12,7)) ;
        f11_21[k]  = (( x11s[k] - x21s[k] ) / d_11_21 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_11_21,13) - pow(sigma_LJ/d_11_21,7)) ;
        f11_22[k]  = (( x11s[k] - x22s[k] ) / d_11_22 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_11_22,13) - pow(sigma_LJ/d_11_22,7)) ;
        f12_21[k]  = (( x12s[k] - x21s[k] ) / d_12_21 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_12_21,13) - pow(sigma_LJ/d_12_21,7)) ;
        f12_22[k]  = (( x12s[k] - x22s[k] ) / d_12_22 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_12_22,13) - pow(sigma_LJ/d_12_22,7)) ;
        f21_22[k]  = (( x21s[k] - x22s[k] ) / d_21_22 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_21_22,13) - pow(sigma_LJ/d_21_22,7)) ;

        f11[k] =  - f11_21[k] - f11_22[k] - f11_12[k];
        f12[k] =  - f12_21[k] - f12_22[k] + f11_12[k];
        f21[k] =  f12_21[k] + f11_21[k] - f21_22[k];
        f22[k] = f12_22[k] + f11_22[k] + f21_22[k];

        v11s[k] = q11[k] + 0.5 * ( f11[k]  / atom_mass_i )* dt;
        v12s[k] = q12[k] + 0.5 * ( f12[k] / atom_mass_i )* dt;
        v21s[k] = q21[k] + 0.5 * ( f21[k] / atom_mass_j )* dt;
        v22s[k] = q22[k] + 0.5 * ( f22[k] / atom_mass_j )* dt;
      }

      // RATTLE part 2: constrain velocities to be perpendicular to bond.
      tol = 1e-16;
      while ( true ) {
        err1 = abs(  (v11s[0] - v12s[0])*(x11s[0] - x12s[0]) + (v11s[1] - v12s[1])*(x11s[1] - x12s[1]) + (v11s[2] - v12s[2])*(x11s[2] - x12s[2]) );
        err2 = abs(  (v21s[0] - v22s[0])*(x21s[0] - x22s[0]) + (v21s[1] - v22s[1])*(x21s[1] - x22s[1]) + (v21s[2] - v22s[2])*(x21s[2] - x22s[2]) );
        if (err2 < tol && err1 < tol ){ 
          break;
        }

        k1 = ((v11s[0] - v12s[0])*(x11s[0] - x12s[0]) + (v11s[1] - v12s[1])*(x11s[1] - x12s[1]) + (v11s[2] - v12s[2])*(x11s[2] - x12s[2])) / (pow( bond_length_i,  2) * (2/atom_mass_i));
        k2 = ((v21s[0] - v22s[0])*(x21s[0] - x22s[0]) + (v21s[1] - v22s[1])*(x21s[1] - x22s[1]) + (v21s[2] - v22s[2])*(x21s[2] - x22s[2]) ) / (pow( bond_length_j,  2) * (2/atom_mass_j));

        for (int k=0;k<3;k++){
          v11s[k] -= (k1 * (x11s[k] - x12s[k]) / atom_mass_i);
          v12s[k] += (k1 * (x11s[k] - x12s[k]) / atom_mass_i);
          v21s[k] -= (k2 * (x21s[k] - x22s[k]) / atom_mass_j);
          v22s[k] += (k2 * (x21s[k] - x22s[k]) / atom_mass_j);
        } 
      }
      
      if ( i>200 && d_11_22>d_initial ){
        break;
      }
    } 
    
    // Calculate new particle internal energies
    double vcm_post_1[3], vcm_post_2[3];

    for (int k=0;k<3;k++){
      vcm_post_1[k] = (v11s[k] + v12s[k])/2;
      vcm_post_2[k] = (v21s[k] + v22s[k])/2;
    }

    double omega1[3], omega2[3];
    
    omega1[0] = ((v11s[1]-vcm_post_1[1])* ((x11s[2] - x12s[2])/2 ) - (v11s[2]-vcm_post_1[2])* ((x11s[1] - x12s[1])/2 ) )/ (pow((x11s[0] - x12s[0])/2,2) + pow((x11s[1] - x12s[1])/2,2) + pow((x11s[2] - x12s[2])/2,2));
    omega1[1] = ((v11s[2]-vcm_post_1[2])* ((x11s[0] - x12s[0])/2 ) - (v11s[0]-vcm_post_1[0])* ((x11s[2] - x12s[2])/2 ) )/ (pow((x11s[0] - x12s[0])/2,2) + pow((x11s[1] - x12s[1])/2,2) + pow((x11s[2] - x12s[2])/2,2));
    omega1[2] = ((v11s[0]-vcm_post_1[0])* ((x11s[1] - x12s[1])/2 ) - (v11s[1]-vcm_post_1[1])* ((x11s[0] - x12s[0])/2 ) )/ (pow((x11s[0] - x12s[0])/2,2) + pow((x11s[1] - x12s[1])/2,2) + pow((x11s[2] - x12s[2])/2,2));
    
    omega2[0] = ((v21s[1]-vcm_post_2[1])* ((x21s[2] - x22s[2])/2 ) - (v21s[2]-vcm_post_2[2])* ((x21s[1] - x22s[1])/2 ) )/ (pow((x21s[0] - x22s[0])/2,2) + pow((x21s[1] - x22s[1])/2,2) + pow((x21s[2] - x22s[2])/2,2));
    omega2[1] = ((v21s[2]-vcm_post_2[2])* ((x21s[0] - x22s[0])/2 ) - (v21s[0]-vcm_post_2[0])* ((x21s[2] - x22s[2])/2 ) )/ (pow((x21s[0] - x22s[0])/2,2) + pow((x21s[1] - x22s[1])/2,2) + pow((x21s[2] - x22s[2])/2,2));
    omega2[2] = ((v21s[0]-vcm_post_2[0])* ((x21s[1] - x22s[1])/2 ) - (v21s[1]-vcm_post_2[1])* ((x21s[0] - x22s[0])/2 ) )/ (pow((x21s[0] - x22s[0])/2,2) + pow((x21s[1] - x22s[1])/2,2) + pow((x21s[2] - x22s[2])/2,2));

    erot1_new = 0.5 * I1 * (pow( omega1[0], 2) + pow(omega1[1], 2) + pow(omega1[2], 2)) ;
    erot2_new = 0.5 * I2 * (pow( omega2[0], 2) + pow(omega2[1], 2) + pow(omega2[2], 2));

    postcoln.etrans = 0.5 * params[isp][jsp].mr * (pow( vcm_post_1[0] - vcm_post_2[0], 2) +pow( vcm_post_1[1] - vcm_post_2[1], 2) +pow( vcm_post_1[2] - vcm_post_2[2], 2) );

    coschi = vcm_post_1[0] / sqrt( pow(vcm_post_1[0],2) +  pow(vcm_post_1[1],2) + pow(vcm_post_1[2],2) );

    if (training && training_data.outputs.size()/training_data.num_outputs < train_params.len_data  ){
        double e_ref = 100;
        double b_ref = 5;
        double e_star = precoln.etrans / (epsilon_LJ * e_ref);
        double b_star = b / (sigma_LJ * b_ref);

        training_data.features.push_back(e_star);
        training_data.features.push_back(b_star);
        training_data.features.push_back(ip->erot/(epsilon_LJ * e_ref));
        training_data.features.push_back(jp->erot/(epsilon_LJ * e_ref));

        training_data.features.push_back(theta1);
        training_data.features.push_back(theta2);
        training_data.features.push_back(phi1);
        training_data.features.push_back(phi2);
        training_data.features.push_back(eta1);
        training_data.features.push_back(eta2);

        training_data.outputs.push_back(acos(coschi) /MY_PI);
        training_data.outputs.push_back(postcoln.etrans/precoln.etotal); // R
        training_data.outputs.push_back(erot1_new / ( erot1_new + erot2_new) ); // r
    }
  } else {

    double e_ref = 100;
    double b_ref = 5;
    double e_star = precoln.etrans / (epsilon_LJ * e_ref);
    double b_star = b / (sigma_LJ * b_ref);
    double input_data[] = {e_star, 
                          b_star, 
                          ip->erot/(epsilon_LJ * e_ref),
                          jp->erot/(epsilon_LJ * e_ref),
                          theta1, theta2, phi1, phi2, eta1, eta2,
                          };
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor inputs = torch::from_blob(input_data, {training_data.num_features}, options);
    torch::Tensor pred = (*CollisionModel).forward(inputs);
    double chi = pred[0].item<double>() * MY_PI;
    coschi = cos( chi );

    double R = pred[1].item<double>();
    double r = pred[2].item<double>();

    postcoln.etrans = R * precoln.etotal;
    erot1_new = r * (1-R) * precoln.etotal;
    erot2_new = (1-r)* (1-R) * precoln.etotal;
  }

  ip->erot = erot1_new;
  jp->erot = erot2_new;

  postcoln.erot = ip->erot + jp->erot;

  double sinchi = sqrt(1-coschi*coschi);
  double eps = random->uniform() * 2*MY_PI;

  double *vi = ip->v;
  double *vj = jp->v;

  vrc[0] = vi[0]-vj[0];
  vrc[1] = vi[1]-vj[1];
  vrc[2] = vi[2]-vj[2];

  double scale = sqrt((2.0 * postcoln.etrans) / (params[isp][jsp].mr * precoln.vr2));
  d = sqrt(vrc[1]*vrc[1]+vrc[2]*vrc[2]);
  if (d > 1.0e-6) {
    ua = scale * ( coschi*vrc[0] + sinchi*d*sin(eps) );
    vb = scale * ( coschi*vrc[1] + sinchi*(precoln.vr*vrc[2]*cos(eps) -
                                        vrc[0]*vrc[1]*sin(eps))/d );
    wc = scale * ( coschi*vrc[2] - sinchi*(precoln.vr*vrc[1]*cos(eps) +
                                        vrc[0]*vrc[2]*sin(eps))/d );
  } else {
    ua = scale * ( coschi*vrc[0] );
    vb = scale * ( sinchi*vrc[0]*cos(eps) );
    wc = scale * ( sinchi*vrc[0]*sin(eps) );
  }

  double divisor = 1.0 / (mass_i + mass_j);
  vi[0] = precoln.ucmf + (mass_j*divisor)*ua;
  vi[1] = precoln.vcmf + (mass_j*divisor)*vb;
  vi[2] = precoln.wcmf + (mass_j*divisor)*wc;
  vj[0] = precoln.ucmf - (mass_i*divisor)*ua;
  vj[1] = precoln.vcmf - (mass_i*divisor)*vb;
  vj[2] = precoln.wcmf - (mass_i*divisor)*wc;
}

/* ----------------------------------------------------------------------
   return a per-species parameter to caller
------------------------------------------------------------------------- */

double CollideDMS::extract(int isp, int jsp, const char *name)
{
  if (strcmp(name,"sigma") == 0) return params[isp][jsp].sigma;
  else if (strcmp(name,"epsilon") == 0) return params[isp][jsp].epsilon;
  else if (strcmp(name,"A") == 0) return params[isp][jsp].A;
    else if (strcmp(name,"B") == 0) return params[isp][jsp].B;
  else if (strcmp(name,"C") == 0) return params[isp][jsp].C;
  else error->all(FLERR,"Request for unknown parameter from collide");
  return 0.0;
}


/* ----------------------------------------------------------------------
   read list of species defined in species file
   store info in filespecies and nfilespecies
   only invoked by proc 0
------------------------------------------------------------------------- */

void CollideDMS::read_param_file(char *fname)
{
  FILE *fp = fopen(fname,"r");
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open DMS parameter file %s",fname);
    error->one(FLERR,str);
  }

  // set all species diameters to -1, so can detect if not read
  // set all cross-species parameters to -1 to catch no-reads, as
  // well as user-selected average

  for (int i = 0; i < nparams; i++) {
    params[i][i].sigma = -1.0;
    for ( int j = i+1; j<nparams; j++) {
      params[i][j].sigma = params[i][j].epsilon = params[i][j].A = -1.0;
      params[i][j].B = params[i][j].C  = -1.0;
      params[i][j].timesteps = params[i][j].dt_verlet = params[i][j].mr = -1.0;
      params[i][j].bond_length_i = params[i][j].bond_length_j =-1.0;
    }
  }

  // read file line by line
  // skip blank lines or comment lines starting with '#'
  // all other lines must have at least REQWORDS, which depends on VARIABLE flag

  int REQWORDS = 9;
  char **words = new char*[REQWORDS+1]; // one extra word in cross-species lines
  char line[MAXLINE];
  int isp,jsp;

  while (fgets(line,MAXLINE,fp)) {
    int pre = strspn(line," \t\n\r");
    if (pre == strlen(line) || line[pre] == '#') continue;

    int nwords = wordparse(REQWORDS+1,line,words);
    if (nwords < REQWORDS)
      error->one(FLERR,"Incorrect line format in DMS parameter file");

    isp = particle->find_species(words[0]);
    if (isp < 0) continue;

    jsp = particle->find_species(words[1]);

    // if we don't match a species with second word, but it's not a number,
    // skip the line (it involves a species we aren't using)
    if ( jsp < 0 &&  !(atof(words[1]) > 0) ) continue;

    if (jsp < 0 ) {
      params[isp][isp].sigma = atof(words[1]);
      params[isp][isp].epsilon = atof(words[2]) * update->boltz;
      params[isp][isp].A = atof(words[3]);
      params[isp][isp].B = atof(words[4]);
      params[isp][isp].C = atof(words[5]);
      params[isp][isp].timesteps = atof(words[6]);
      params[isp][isp].dt_verlet = atof(words[7]);
      params[isp][isp].bond_length_i = atof(words[8]);
      params[isp][isp].bond_length_j = atof(words[8]);
    } else {
      if (nwords < REQWORDS+2)  // one extra word in cross-species lines
        error->one(FLERR,"Incorrect line format in DMS parameter file");
      params[isp][jsp].sigma = params[jsp][isp].sigma = atof(words[2]);
      params[isp][jsp].epsilon = params[jsp][isp].epsilon = atof(words[3]) * update->boltz;
      params[isp][jsp].A = params[jsp][isp].A = atof(words[4]);
      params[isp][jsp].B = params[jsp][isp].B = atof(words[5]);
      params[isp][jsp].C = params[jsp][isp].C = atof(words[6]);
      params[isp][jsp].timesteps = params[jsp][isp].timesteps = atof(words[7]);
      params[isp][jsp].dt_verlet = params[jsp][isp].dt_verlet = atof(words[8]);
      params[isp][jsp].bond_length_i = params[jsp][isp].bond_length_j = atof(words[9]); // This should never be used?
      params[isp][jsp].bond_length_j = params[jsp][isp].bond_length_i = atof(words[10]);
    }
  }

  delete [] words;
  fclose(fp);

  // check that params were read for all species
  for (int i = 0; i < nparams; i++) {

    if (params[i][i].sigma < 0.0) {
      char str[128];
      sprintf(str,"Species %s did not appear in DMS parameter file",
              particle->species[i].id);
      error->one(FLERR,str);
    }
  }

  for ( int i = 0; i<nparams; i++) {
    params[i][i].mr = particle->species[i].mass / 2;
    for ( int j = i+1; j<nparams; j++) {
      params[i][j].mr = params[j][i].mr = particle->species[i].mass *
        particle->species[j].mass / (particle->species[i].mass + particle->species[j].mass);

      if(params[i][j].sigma < 0) params[i][j].sigma = params[j][i].sigma =
                                  0.5*(params[i][i].sigma + params[j][j].sigma);
      if(params[i][j].epsilon < 0) params[i][j].epsilon = params[j][i].epsilon =
                                   0.5*(params[i][i].epsilon + params[j][j].epsilon);
      if(params[i][j].A < 0) params[i][j].A = params[j][i].A =
                                  MAX(params[i][i].A, params[j][j].A);
      if(params[i][j].B < 0) params[i][j].B = params[j][i].B =
                                   MAX(params[i][i].B, params[j][j].B);
      if(params[i][j].C < 0) params[i][j].C = params[j][i].C =
                                  MAX(params[i][i].C, params[j][j].C);
      if(params[i][j].dt_verlet < 0) params[i][j].dt_verlet = params[j][i].dt_verlet =
                                   MIN(params[i][i].dt_verlet, params[j][j].dt_verlet);
      if(params[i][j].timesteps < 0) params[i][j].timesteps = params[j][i].timesteps =
                                  MAX(params[i][i].timesteps, params[j][j].timesteps);
      if(params[i][j].bond_length_i < 0) params[i][j].bond_length_i = params[j][i].bond_length_j =
                                  params[i][i].bond_length_i;
      if(params[i][j].bond_length_j < 0) params[i][j].bond_length_j = params[j][i].bond_length_i =
                                  params[j][j].bond_length_j;
    }
  }
}

/* ----------------------------------------------------------------------
   parse up to n=maxwords whitespace-delimited words in line
   store ptr to each word in words and count number of words
------------------------------------------------------------------------- */

int CollideDMS::wordparse(int maxwords, char *line, char **words)
{
  int nwords = 1;
  char * word;

  words[0] = strtok(line," \t\n");
  while ((word = strtok(NULL," \t\n")) != NULL && nwords < maxwords) {
    words[nwords++] = word;
  }
  return nwords;
}
