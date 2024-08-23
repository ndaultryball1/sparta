#include "collide_dms.h"
#include "math_const.h"
#include "comm.h"

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