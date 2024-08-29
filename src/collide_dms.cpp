#include "collide_dms.h"
#include "math_const.h"
#include "comm.h"
#include "math.h"
#include "random_knuth.h"
#include "mixture.h"

using namespace SPARTA_NS;
using namespace MathConst;

CollideDMS::CollideDMS(SPARTA *sparta, int narg, char **arg) :
  Collide(sparta,narg,arg)
{ 
  // Processing for optional args as in VSS?

  // As in collide_vss.cpp
  nparams = particle->nspecies;
  if (nparams == 0)
    error->all(FLERR,"Cannot use collide command with no species defined");

  memory->create(params,nparams,nparams,"collide:params");
  if (comm->me == 0) read_param_file(arg[2]);
  MPI_Bcast(params[0],nparams*nparams*sizeof(Params),MPI_BYTE,0,world);

  // allocate per-species prefactor array

  memory->create(prefactor,nparams,nparams,"collide:prefactor");
}

CollideDMS::~CollideDMS()
{
  if (copymode) return;

  memory->destroy(params);
  memory->destroy(prefactor);
}

void CollideDMS::init()
{
  // initially read-in per-species params must match current species list

  if (nparams != particle->nspecies)
    error->all(FLERR,"DMS parameters do not match current species");

  Collide::init();
}

double CollideDMS::b_max(Particle::OnePart *ip, Particle::OnePart *jp)
{ 
  int isp = ip->ispecies;
  int jsp = jp->ispecies;
  return params[isp][jsp].sigma * ( pow( params[isp][jsp].A * precoln.vr, params[isp][jsp].B ) + params[isp][jsp].C );
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

      double cxs = params[isp][jsp].sigma*params[isp][jsp].sigma*MY_PI; // This is not a good estimate
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
  double vro  = pow(vr2,0.5); // Equal to v_rel ** (1-2*nu)

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

  double imass = precoln.imass = species[isp].mass;
  double jmass = precoln.jmass = species[jsp].mass;

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
  reactflag =0;
  if (react) // raise some kind of error?
    ;
  else
    reactflag=0;
    if (precoln.erot=0.)
    {
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
  double *vi = ip->v;
  double *vj = jp->v;
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
  double dist, d;
  double f1[3], f2, x2s[3], x1s[3], v1s[3], v2s[3];

  x2s[0] = precoln.D_cutoff;
  x2s[1] = pow(random->uniform(), 0.5) * precoln.bmax;
  
  v1s[0] = precoln.vr;

  double v_cm_x = (v1s[0]+ v2s[0] ) / 2;
  double v_cm_y = (v1s[1]+ v2s[1] ) / 2; 
  v1s[0] = v1s[0] - v_cm_x;
  v2s[0] = v2s[0] - v_cm_x;
  v1s[1] = v1s[1] - v_cm_y;
  v2s[1] = v2s[1] - v_cm_y;


  for (int i=0;i<params[isp][jsp].timesteps;i++){
    
    dist = sqrt(  pow( x1s[0]-x2s[0], 2) + pow(x1s[1]-x2s[1], 2) + pow(x1s[2]-x2s[2], 2) );
    d = sigma_LJ / dist;
    for (int k=0;k<3;k++){
      f1[k]  = (( x1s[k] - x2s[k] ) / dist )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(d,13) - pow(d,7)) ;
      x1s[k] = x1s[k] + v1s[k] * dt - 0.5 * ( f1[k]/mass_i ) * pow(dt,2);
      x2s[k] = x2s[k] + v2s[k] * dt + 0.5 * ( f1[k]/mass_j ) * pow(dt,2);
    }

    dist = sqrt(  pow( x1s[0]-x2s[0], 2) + pow(x1s[1]-x2s[1], 2) + pow(x1s[2]-x2s[2], 2) );

    d = sigma_LJ / dist;
    for (int k=0;k<3;k++){
      f2 = (( x1s[k] - x2s[k] ) / dist )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(d,13) - pow(d,7));
      v1s[k] = v1s[k] - 0.5 * ( f1[k] + f2 ) / mass_i * dt;
      v2s[k] = v2s[k] + 0.5 * ( f1[k] + f2 ) / mass_j * dt;
    }
  }
  double coschi = v1s[0] / sqrt( pow(v1s[0],2) +  pow(v1s[1],2) + pow(v1s[2],2) );
  double sinchi = v1s[1] / sqrt( pow(v1s[0],2) +  pow(v1s[1],2) + pow(v1s[2],2) );
  double eps = random->uniform() * 2*MY_PI;

  vrc[0] = vi[0]-vj[0];
  vrc[1] = vi[1]-vj[1];
  vrc[2] = vi[2]-vj[2];
  
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