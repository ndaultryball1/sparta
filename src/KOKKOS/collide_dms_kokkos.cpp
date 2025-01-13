/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "collide_dms_kokkos.h"
#include "grid.h"
#include "update.h"
#include "particle_kokkos.h"
#include "mixture.h"
#include "collide.h"
#include "react.h"
#include "comm.h"
#include "random_knuth.h"
#include "random_mars.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "error.h"
#include "kokkos.h"
#include "sparta_masks.h"
#include "modify.h"
#include "fix.h"
#include "fix_ambipolar.h"

using namespace SPARTA_NS;
using namespace MathConst;

enum{NONE,DISCRETE,SMOOTH};            // several files
enum{CONSTANT,VARIABLE};

#define DELTAGRID 1000            // must be bigger than split cells per cell
#define DELTADELETE 1024
#define DELTAELECTRON 128
#define DELTACELLCOUNT 2

#define MAXLINE 1024
#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

CollideDMSKokkos::CollideDMSKokkos(SPARTA *sparta, int narg, char **arg) :
  CollideDMS(sparta, narg, arg),
  rand_pool(12345 + comm->me
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            ),
  grid_kk_copy(sparta),
  react_kk_copy(sparta)
{
  kokkos_flag = 1;

  // use 1D view for scalars to reduce GPU memory operations

  d_scalars = t_int_11("collide:scalars");
  h_scalars = t_host_int_11("collide:scalars_mirror");

  d_nattempt_one = Kokkos::subview(d_scalars,0);
  d_ncollide_one = Kokkos::subview(d_scalars,1);
  d_nreact_one   = Kokkos::subview(d_scalars,2);
  d_error_flag   = Kokkos::subview(d_scalars,3);
  d_retry        = Kokkos::subview(d_scalars,4);
  d_maxdelete    = Kokkos::subview(d_scalars,5);
  d_maxcellcount = Kokkos::subview(d_scalars,6);
  d_part_grow    = Kokkos::subview(d_scalars,7);
  d_ndelete      = Kokkos::subview(d_scalars,8);
  d_nlocal       = Kokkos::subview(d_scalars,9);
  d_maxelectron  = Kokkos::subview(d_scalars,10);

  h_nattempt_one = Kokkos::subview(h_scalars,0);
  h_ncollide_one = Kokkos::subview(h_scalars,1);
  h_nreact_one   = Kokkos::subview(h_scalars,2);
  h_error_flag   = Kokkos::subview(h_scalars,3);
  h_retry        = Kokkos::subview(h_scalars,4);
  h_maxdelete    = Kokkos::subview(h_scalars,5);
  h_maxcellcount = Kokkos::subview(h_scalars,6);
  h_part_grow    = Kokkos::subview(h_scalars,7);
  h_ndelete      = Kokkos::subview(h_scalars,8);
  h_nlocal       = Kokkos::subview(h_scalars,9);
  h_maxelectron  = Kokkos::subview(h_scalars,10);

  random_backup = NULL;
  react_defined = 0;

  maxdelete = DELTADELETE;
}

/* ---------------------------------------------------------------------- */

CollideDMSKokkos::~CollideDMSKokkos()
{
  if (copymode) return;

  grid_kk_copy.uncopy();
  react_kk_copy.uncopy();

  memoryKK->destroy_kokkos(k_dellist,dellist);

#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
  if (random_backup)
    delete random_backup;
#endif
}

/* ---------------------------------------------------------------------- */

void CollideDMSKokkos::init()
{
  // error check

  // initially read-in per-species params must match current species list

  if (nparams != particle->nspecies)
    error->all(FLERR,"DMS parameters do not match current species");

  // require mixture to contain all species

  int imix = particle->find_mixture(mixID);
  if (imix < 0) error->all(FLERR,"Collision mixture does not exist");
  mixture = particle->mixture[imix];

  if (mixture->nspecies != particle->nspecies)
    error->all(FLERR,"Collision mixture does not contain all species");

  // if rotstyle or vibstyle = DISCRETE,
  // check that extra rotation/vibration info is defined
  // for species that require it

  if (vibstyle == DISCRETE) {
    index_vibmode = particle->find_custom((char *) "vibmode");

    Particle::Species *species = particle->species;
    int nspecies = particle->nspecies;

    int flag = 0;
    for (int isp = 0; isp < nspecies; isp++) {
      if (species[isp].vibdof <= 2) continue;
      if (index_vibmode < 0)
        error->all(FLERR,
                   "Fix vibmode must be used with discrete vibrational modes");
      if (species[isp].nvibmode != species[isp].vibdof / 2) flag++;
    }
    if (flag) {
      char str[128];
      sprintf(str,"%d species do not define correct vibrational "
              "modes for discrete model",flag);
      error->all(FLERR,str);
    }
  }

  // reallocate one-cell data structs for one or many groups

  oldgroups = ngroups;
  ngroups = mixture->ngroup;

  if (ngroups != oldgroups) {
    if (oldgroups == 1) {
      memory->destroy(plist);
      npmax = 0;
      plist = NULL;
    }
    if (oldgroups > 1) {
      delete [] ngroup;
      delete [] maxgroup;
      for (int i = 0; i < oldgroups; i++) memory->destroy(glist[i]);
      delete [] glist;
      memory->destroy(gpair);
      ngroup = NULL;
      maxgroup = NULL;
      glist = NULL;
      gpair = NULL;
    }

    if (ngroups == 1) {
      npmax = DELTAPART;
      memory->create(plist,npmax,"collide:plist");
    }
    if (ngroups > 1) {
      ngroup = new int[ngroups];
      maxgroup = new int[ngroups];
      glist = new int*[ngroups];
      for (int i = 0; i < ngroups; i++) {
        maxgroup[i] = DELTAPART;
        memory->create(glist[i],DELTAPART,"collide:glist");
      }
      memory->create(gpair,ngroups*ngroups,3,"collide:gpair");
    }
  }

  // allocate vremax,remain if group count changed
  // will always be allocated on first run since oldgroups = 0
  // set vremax_intitial via values calculated by collide style

  if (ngroups != oldgroups) {
    memory->destroy(vremax_initial);
    nglocal = grid->nlocal;
    nglocalmax = nglocal;
    memory->create(vremax_initial,ngroups,ngroups,"collide:vremax_initial");

    k_vremax_initial = DAT::tdual_float_2d("collide:vremax_initial",ngroups,ngroups);
    k_vremax = DAT::tdual_float_3d("collide:vremax",nglocalmax,ngroups,ngroups);
    d_vremax = k_vremax.d_view;
    k_remain = DAT::tdual_float_3d("collide:remain",nglocalmax,ngroups,ngroups);
    d_remain = k_remain.d_view;

    for (int igroup = 0; igroup < ngroups; igroup++) {
      for (int jgroup = 0; jgroup < ngroups; jgroup++) {
        vremax_initial[igroup][jgroup] = vremax_init(igroup,jgroup);
        k_vremax_initial.h_view(igroup,jgroup) = vremax_initial[igroup][jgroup];
      }
    }

    k_vremax_initial.modify_host();
    k_vremax_initial.sync_device();
    d_vremax_initial = k_vremax_initial.d_view;
  }

  // if recombination reactions exist, set flags per species pair

  recombflag = 0;
  if (react) {
    react_defined = 1;
    recombflag = react->recombflag;
    recomb_boost_inverse = react->recomb_boost_inverse;
  }

  if (recombflag) {
    int nspecies = particle->nspecies;
    //memory->destroy(recomb_ijflag);
    //memory->create(recomb_ijflag,nspecies,nspecies,"collide:recomb_ijflag");
    d_recomb_ijflag = DAT::t_float_2d("collide:recomb_ijflag",nspecies,nspecies);
    auto h_recomb_ijflag = Kokkos::create_mirror_view(d_recomb_ijflag);
    for (int i = 0; i < nspecies; i++)
      for (int j = 0; j < nspecies; j++)
        h_recomb_ijflag(i,j) = react->recomb_exist(i,j);
    Kokkos::deep_copy(d_recomb_ijflag,h_recomb_ijflag);
  }

  // find ambipolar fix
  // set ambipolar vector/array indices
  // if reactions defined, check that they are valid ambipolar reactions

  if (ambiflag) {
    error->all(FLERR,"Ambipolar collisions not supported with Kokkos");
  }

  // if ambipolar and multiple groups in mixture, ambispecies must be its own group

  if (ambiflag && mixture->ngroup > 1) {
    error->all(FLERR,"Ambipolar collisions not supported with Kokkos");
  }

  // vre_next = next timestep to zero vremax & remain, based on vre_every

  if (vre_every) vre_next = (update->ntimestep/vre_every)*vre_every + vre_every;
  else vre_next = update->laststep + 1;

  // if requested reset vremax & remain
  // must be after per-species vremax_initial is setup

  if (vre_first || vre_start) {
    reset_vremax();
    vre_first = 0;
  }

#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.init(random);
#endif

  // DMS specific

  k_params = tdual_params_2d("collide_dms:params",nparams,nparams);

  for (int i = 0; i < nparams; i++) {
    for (int j = 0; j < nparams; j++){
      k_params.h_view(i,j) = params[i][j];
    }
  }

  k_params.modify_host();
  k_params.sync_device();
  d_params = k_params.d_view;

  // initialize running stats before each run

  ncollide_running = nattempt_running = nreact_running = 0;
}

/* ----------------------------------------------------------------------
   reset vremax to initial species-based values
   reset remain to 0.0
------------------------------------------------------------------------- */

void CollideDMSKokkos::reset_vremax()
{
  grid_kk_copy.copy((GridKokkos*)grid);

  this->sync(Device,ALL_MASK);

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCollideResetVremax>(0,nglocal),*this);
  copymode = 0;

  this->modified(Device,ALL_MASK);
}

KOKKOS_INLINE_FUNCTION
void CollideDMSKokkos::operator()(TagCollideResetVremax, const int &icell) const {
  for (int igroup = 0; igroup < ngroups; igroup++)
    for (int jgroup = 0; jgroup < ngroups; jgroup++) {
      d_vremax(icell,igroup,jgroup) = d_vremax_initial(igroup,jgroup);
      if (remainflag) d_remain(icell,igroup,jgroup) = 0.0;
    }
}

/* ----------------------------------------------------------------------
  NTC algorithm
------------------------------------------------------------------------- */

void CollideDMSKokkos::collisions()
{
  // if requested, reset vrwmax & remain

  if (update->ntimestep == vre_next) {
    reset_vremax();
    vre_next += vre_every;
  }

  // counters

  ncollide_one = nattempt_one = nreact_one = 0;
  h_ndelete() = 0;

  if (sparta->kokkos->atomic_reduction) {
    h_nattempt_one() = 0;
    h_ncollide_one() = 0;
    h_nreact_one() = 0;
  }

  dt = update->dt;
  fnum = update->fnum;
  boltz = update->boltz;

  // perform collisions:
  // variant for single group or multiple groups (not yet supported)
  // variant for nearcp flag or not
  // variant for ambipolar approximation or not

  if (ngroups != 1)
    error->all(FLERR,"Group collisions not yet supported with Kokkos");

  COLLIDE_REDUCE reduce;

  if (!ambiflag) {
    if (nearcp == 0)
      collisions_one<0>(reduce);
    else
      collisions_one<1>(reduce);
  } else {
    error->all(FLERR,"Ambipolar not supported with DMS");
  }

  // remove any particles deleted in chemistry reactions
  // if particles deleted/created by chemistry, particles are no longer sorted

  if (ndelete) {
    k_dellist.modify_device();
    k_dellist.sync_host();
    ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
#ifndef SPARTA_KOKKOS_EXACT
    particle_kk->compress_migrate(ndelete,dellist);
#else
    particle->compress_reactions(ndelete,dellist);
#endif
  }
  if (react) {
    particle->sorted = 0;
    ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
    particle_kk->sorted_kk = 0;
  }

  // accumulate running totals

  if (sparta->kokkos->atomic_reduction) {
    nattempt_one = h_nattempt_one();
    ncollide_one = h_ncollide_one();
    nreact_one = h_nreact_one();
  } else {
    nattempt_one += reduce.nattempt_one;
    ncollide_one += reduce.ncollide_one;
    nreact_one += reduce.nreact_one;
  }

  nattempt_running += nattempt_one;
  ncollide_running += ncollide_one;
  nreact_running += nreact_one;
}

/* ----------------------------------------------------------------------
   NTC algorithm for a single group
------------------------------------------------------------------------- */

template < int NEARCP > void CollideDMSKokkos::collisions_one(COLLIDE_REDUCE &reduce)
{
  // loop over cells I own

  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK);
  if (vibstyle == DISCRETE) particle_kk->sync(Device,CUSTOM_MASK);
  d_particles = particle_kk->k_particles.d_view;
  d_species = particle_kk->k_species.d_view;
  d_ewhich = particle_kk->k_ewhich.d_view;
  k_eiarray = particle_kk->k_eiarray;

  GridKokkos* grid_kk = (GridKokkos*) grid;
  grid_kk->sync(Device,CINFO_MASK);
  d_plist = grid_kk->d_plist;

  grid_kk_copy.copy(grid_kk);

  if (react) {
    ReactTCEKokkos* react_kk = (ReactTCEKokkos*) react;
    if (!react_kk)
      error->all(FLERR,"Must use TCE reactions with Kokkos");
    react_kk_copy.copy(react_kk);
  }

  copymode = 1;

  if (NEARCP) {
    error->all(FLERR,"Nearest neighbour collisions not supported with DMS");
  }

  /* ATOMIC_REDUCTION: 1 = use atomics
                       0 = don't need atomics
                      -1 = use parallel_reduce
  */

  // Reactions may create or delete more particles than existing views can hold.
  //  Cannot grow a Kokkos view in a parallel loop, so
  //  if the capacity of the view is exceeded, break out of parallel loop,
  //  reallocate on the host, and then repeat the parallel loop again.
  //  Unfortunately this leads to really messy code.

  h_retry() = 1;

  double extra_factor = sparta->kokkos->collide_extra;
  if (sparta->kokkos->collide_retry_flag) extra_factor = 1.0;

  if (react) {
    auto maxdelete_extra = maxdelete*extra_factor;
    if (d_dellist.extent(0) < maxdelete_extra) {
      memoryKK->destroy_kokkos(k_dellist,dellist);
      memoryKK->create_kokkos(k_dellist,dellist,maxdelete_extra,"collide:dellist");
      d_dellist = k_dellist.d_view;
    }

    maxcellcount = particle_kk->get_maxcellcount();
    auto maxcellcount_extra = maxcellcount*extra_factor;
    if (d_plist.extent(1) < maxcellcount_extra) {
      d_plist = decltype(d_plist)();
      Kokkos::resize(grid_kk->d_plist,nglocal,maxcellcount_extra);
      d_plist = grid_kk->d_plist;
      if (NEARCP)
        error->all(FLERR,"Nearest neighbour collisions not supported with DMS");
    }

    auto nlocal_extra = particle->nlocal*extra_factor;
    if (d_particles.extent(0) < nlocal_extra) {
      particle->grow(nlocal_extra - particle->nlocal);
      d_particles = particle_kk->k_particles.d_view;
      k_eiarray = particle_kk->k_eiarray;
    }
  }

  while (h_retry()) {

    if (react && sparta->kokkos->collide_retry_flag)
      backup();

    h_retry() = 0;
    h_maxdelete() = maxdelete;
    h_maxcellcount() = maxcellcount;
    h_part_grow() = 0;
    h_ndelete() = 0;
    h_nlocal() = particle->nlocal;

    Kokkos::deep_copy(d_scalars,h_scalars);

    if (sparta->kokkos->atomic_reduction) {
      if (sparta->kokkos->need_atomics)
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCollideCollisionsOne<NEARCP,1> >(0,nglocal),*this);
      else
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCollideCollisionsOne<NEARCP,0> >(0,nglocal),*this);
    } else
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagCollideCollisionsOne<NEARCP,-1> >(0,nglocal),*this,reduce);

    Kokkos::deep_copy(h_scalars,d_scalars);

    if (h_retry()) {
      //printf("Retrying, reason %i %i %i !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",h_maxdelete() > d_dellist.extent(0),h_maxcellcount() > d_plist.extent(1),h_part_grow());
      if (!sparta->kokkos->collide_retry_flag) {
        error->one(FLERR,"Ran out of space in Kokkos collisions, increase collide/extra"
                         " or use collide/retry");
      } else
        restore();

      reduce = COLLIDE_REDUCE();

      maxdelete = h_maxdelete();
      auto maxdelete_extra = maxdelete*extra_factor;
      if (d_dellist.extent(0) < maxdelete_extra) {
        memoryKK->destroy_kokkos(k_dellist,dellist);
        memoryKK->grow_kokkos(k_dellist,dellist,maxdelete_extra,"collide:dellist");
        d_dellist = k_dellist.d_view;
      }

      maxcellcount = h_maxcellcount();
      particle_kk->set_maxcellcount(maxcellcount);
      auto maxcellcount_extra = maxcellcount*extra_factor;
      if (d_plist.extent(1) < maxcellcount_extra) {
        d_plist = decltype(d_plist)();
        Kokkos::resize(grid_kk->d_plist,nglocal,maxcellcount_extra);
        d_plist = grid_kk->d_plist;
      }

      auto nlocal_extra = h_nlocal()*extra_factor;
      if (d_particles.extent(0) < nlocal_extra) {
        particle->grow(nlocal_extra - particle->nlocal);
        d_particles = particle_kk->k_particles.d_view;
        k_eiarray = particle_kk->k_eiarray;
      }
    }
  }

  ndelete = h_ndelete();

  particle->nlocal = h_nlocal();

  copymode = 0;

  if (h_error_flag())
    error->one(FLERR,"Collision cell volume is zero");

  particle_kk->modify(Device,PARTICLE_MASK);

  d_particles = t_particle_1d(); // destroy reference to reduce memory use
  d_nn_last_partner = decltype(d_nn_last_partner)();
  d_plist = decltype(d_nn_last_partner)();
}

KOKKOS_INLINE_FUNCTION
void CollideDMSKokkos::operator()(TagCollideZeroNN, const int &icell) const {
  const int np = grid_kk_copy.obj.d_cellcount[icell];
  for (int i = 0; i < np; i++)
    d_nn_last_partner(icell,i) = 0;
}

template < int NEARCP, int ATOMIC_REDUCTION >
KOKKOS_INLINE_FUNCTION
void CollideDMSKokkos::operator()(TagCollideCollisionsOne< NEARCP, ATOMIC_REDUCTION >, const int &icell) const {
  COLLIDE_REDUCE reduce;
  this->template operator()< NEARCP, ATOMIC_REDUCTION >(TagCollideCollisionsOne< NEARCP, ATOMIC_REDUCTION >(), icell, reduce);
}

template < int NEARCP, int ATOMIC_REDUCTION >
KOKKOS_INLINE_FUNCTION
void CollideDMSKokkos::operator()(TagCollideCollisionsOne< NEARCP, ATOMIC_REDUCTION >, const int &icell, COLLIDE_REDUCE &reduce) const {
  if (d_retry()) return;

  int np = grid_kk_copy.obj.d_cellcount[icell];
  if (np <= 1) return;

  if (NEARCP) {
    error->all(FLERR,"Nearest neighbour collisions not supported with DMS");
  }

  const double volume = grid_kk_copy.obj.k_cinfo.d_view[icell].volume / grid_kk_copy.obj.k_cinfo.d_view[icell].weight;
  if (volume == 0.0) d_error_flag() = 1;

  struct State precoln;       // state before collision
  struct State postcoln;      // state after collision

  rand_type rand_gen = rand_pool.get_state();

  // attempt = exact collision attempt count for a pair of groups
  // nattempt = rounded attempt with RN

  const double attempt = attempt_collision_kokkos(icell,np,volume,rand_gen);
  const int nattempt = static_cast<int> (attempt);
  if (!nattempt){
    rand_pool.free_state(rand_gen);
    return;
  }
  if (ATOMIC_REDUCTION == 1)
    Kokkos::atomic_add(&d_nattempt_one(),nattempt);
  else if (ATOMIC_REDUCTION == 0)
    d_nattempt_one() += nattempt;
  else
    reduce.nattempt_one += nattempt;

  // perform collisions
  // select random pair of particles, cannot be same
  // test if collision actually occurs

  for (int m = 0; m < nattempt; m++) {
    const int i = np * rand_gen.drand();
    int j;
    if (NEARCP) error->all(FLERR,"Nearest neighbour collisions not supported with DMS");
    else {
      j = np * rand_gen.drand();
      while (i == j) j = np * rand_gen.drand();
    }

    Particle::OnePart* ipart = &d_particles[d_plist(icell,i)];
    Particle::OnePart* jpart = &d_particles[d_plist(icell,j)];
    Particle::OnePart* kpart;

    // test if collision actually occurs, then perform it
    // ijspecies = species before collision chemistry
    // continue to next collision if no reaction

    if (!test_collision_kokkos(icell,0,0,ipart,jpart,precoln,rand_gen)) continue;

    if (NEARCP) {
      error->all(FLERR,"Nearest neighbour collisions not supported with DMS");
    }

    // if recombination reaction is possible for this IJ pair
    // pick a 3rd particle to participate and set cell number density
    // unless boost factor turns it off, or there is no 3rd particle

    Particle::OnePart* recomb_part3 = NULL;
    int recomb_species = -1;
    double recomb_density = 0.0;
    if (recombflag && d_recomb_ijflag(ipart->ispecies,jpart->ispecies)) {
      if (rand_gen.drand() > recomb_boost_inverse)
        //react->recomb_species = -1;
        recomb_species = -1;
      else if (np <= 2)
        //react->recomb_species = -1;
        recomb_species = -1;
      else {
        int k = np * rand_gen.drand();
        while (k == i || k == j) k = np * rand_gen.drand();
        // NOT thread safe
        //react->recomb_part3 = &particles[plist[k]];
        //react->recomb_species = react->recomb_part3->ispecies;
        //react->recomb_density = np * update->fnum / volume;
        recomb_part3 = &d_particles[d_plist(icell,k)];
        recomb_species = recomb_part3->ispecies;
        recomb_density = np * fnum / volume;
      }
    }

    // perform collision and possible reaction

    int index_kpart;

    setup_collision_kokkos(ipart,jpart,precoln,postcoln);
    const int reactflag = perform_collision_kokkos(ipart,jpart,kpart,precoln,postcoln,rand_gen,
                                                   recomb_part3,recomb_species,recomb_density,index_kpart);

    if (ATOMIC_REDUCTION == 1)
      Kokkos::atomic_increment(&d_ncollide_one());
    else if (ATOMIC_REDUCTION == 0)
      d_ncollide_one()++;
    else
      reduce.ncollide_one++;

    if (reactflag) {
      if (ATOMIC_REDUCTION == 1)
        Kokkos::atomic_increment(&d_nreact_one());
      else if (ATOMIC_REDUCTION == 0)
        d_nreact_one()++;
      else
        reduce.nreact_one++;
    } else {
      rand_pool.free_state(rand_gen);
      continue;
    }

    // if jpart destroyed, delete from plist
    // also add particle to deletion list
    // exit attempt loop if only single particle left

    if (!jpart) {
      int ndelete = Kokkos::atomic_fetch_add(&d_ndelete(),1);
      if (ndelete < d_dellist.extent(0)) {
        d_dellist(ndelete) = d_plist(icell,j);
      } else {
        d_retry() = 1;
        d_maxdelete() += DELTADELETE;
        rand_pool.free_state(rand_gen);
        return;
      }
      np--;
      d_plist(icell,j) = d_plist(icell,np);
      if (NEARCP) error->all(FLERR,"Nearest neighbour collisions not supported with DMS");
      if (np < 2) break;
    }

    // if kpart created, add to plist
    // kpart was just added to particle list, so index = nlocal-1
    // particle data structs may have been realloced by kpart

    if (kpart) {
      if (np < d_plist.extent(1)) {
        if (NEARCP) error->all(FLERR,"Nearest neighbour collisions not supported with DMS");
        d_plist(icell,np++) = index_kpart;
      } else {
        d_retry() = 1;
        d_maxcellcount() += DELTACELLCOUNT;
        rand_pool.free_state(rand_gen);
        return;
      }

    }
  }
  rand_pool.free_state(rand_gen);
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double CollideDMSKokkos::attempt_collision_kokkos(int icell, int np, double volume, rand_type &rand_gen) const
{
 double nattempt;
 //printf("Reached attempt collision\n");

 if (remainflag) {
   nattempt = 0.5 * np * (np-1) *
     d_vremax(icell,0,0) * dt * fnum / volume + d_remain(icell,0,0);
   d_remain(icell,0,0) = nattempt - static_cast<int> (nattempt);
 } else {
   nattempt = 0.5 * np * (np-1) *
     d_vremax(icell,0,0) * dt * fnum / volume + rand_gen.drand();
 }

 // DEBUG
 //nattempt = 10;

  return nattempt;
}

/* ----------------------------------------------------------------------
   determine if collision actually occurs
   1 = yes, 0 = no
   update vremax either way
------------------------------------------------------------------------- */

int CollideDMSKokkos::test_collision_kokkos(int icell, int igroup, int jgroup,
                                     Particle::OnePart *ip, Particle::OnePart *jp,
                                     struct State &precoln, rand_type &rand_gen) const
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
  // printf("Reached test collision\n");
  double b = (d_params(ispecies,jspecies).A * 
    pow( vro, d_params(ispecies,jspecies).B ) + d_params(ispecies,jspecies).C) 
    * d_params(ispecies,jspecies).sigma;
  precoln.bmax = b;
  double vre = vro*b*b*MY_PI;
  d_vremax(icell,igroup,jgroup) = MAX(vre,d_vremax(icell,igroup,jgroup));
  if (vre/d_vremax(icell,igroup,jgroup) < rand_gen.drand()) return 0;
  precoln.vr2 = vr2;
  return 1;
}


KOKKOS_INLINE_FUNCTION
int CollideDMSKokkos::perform_collision_kokkos(Particle::OnePart *&ip,
                                  Particle::OnePart *&jp,
                                  Particle::OnePart *&kp,
                                  struct State &precoln, struct State &postcoln, rand_type &rand_gen,
                                  Particle::OnePart *&p3, int &recomb_species, double &recomb_density,
                                  int &index_kpart) const
{
  int reactflag;
  if (react)
    error->one(FLERR,"Reaction chemistry not implemented for DMS collision");
  else
    reactflag=0;
    // printf("Reached above choice\n");
    if (precoln.ave_vibdof > 0.0 ) {
      error->all(FLERR,"Scattering not implemented for vibrating molecules.");
      SCATTER_VibDiatomicScatter(ip,jp,precoln,postcoln,rand_gen);
    } else if (precoln.ave_rotdof >0.0 ) {
      // printf("Reached choice\n");
      SCATTER_RigidDiatomicScatter(ip,jp,precoln,postcoln,rand_gen);
    } else {
      SCATTER_MonatomicScatter(ip,jp,precoln,postcoln,rand_gen);
    }


  return reactflag;
}
/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void CollideDMSKokkos::setup_collision_kokkos(Particle::OnePart *ip, Particle::OnePart *jp,
                                       struct State &precoln, struct State &postcoln) const
{
  int isp = ip->ispecies;
  int jsp = jp->ispecies;

  precoln.vr = sqrt(precoln.vr2);

  precoln.ave_rotdof = 0.5 * (d_species[isp].rotdof + d_species[jsp].rotdof);
  precoln.ave_vibdof = 0.5 * (d_species[isp].vibdof + d_species[jsp].vibdof);
  precoln.ave_dof = (precoln.ave_rotdof  + precoln.ave_vibdof)/2.;

  double imass = precoln.imass = d_species[isp].mass;
  double jmass = precoln.jmass = d_species[jsp].mass;

  precoln.etrans = 0.5 * d_params(isp,jsp).mr * precoln.vr2;
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

  precoln.D_cutoff = MAX( 4*d_params(isp,jsp).sigma, 1.5*precoln.bmax);
}

/* ---------------------------------------------------------------------- */
void CollideDMSKokkos::SCATTER_VibDiatomicScatter(Particle::OnePart *ip,
                                          Particle::OnePart *jp,
                                          struct State &precoln, struct State &postcoln,
                                          rand_type &rand_gen) const
{  } 

void CollideDMSKokkos::SCATTER_RigidDiatomicScatter(Particle::OnePart *ip,
                                          Particle::OnePart *jp,
                                          struct State &precoln, struct State &postcoln,
                                          rand_type &rand_gen) const
{   
  // printf("Reached collision again\n");
  // If we have arrived here assume two diatomic molecules. 
  int isp = ip->ispecies;
  int jsp = jp->ispecies;
  double mass_i = d_species[isp].mass;
  double mass_j = d_species[jsp].mass;

  double bond_length_i =  d_params(isp,jsp).bond_length_i; 
  double bond_length_j =  d_params(isp,jsp).bond_length_j;

  // The two atomic masses within a molecule must at the moment be the same i.e. N2, O2.
  double atom_mass_i = mass_i/2;
  double atom_mass_j = mass_j/2;

  double I1 = atom_mass_i/2 * pow( bond_length_i, 2);
  double I2 = atom_mass_j/2 * pow( bond_length_j, 2);

  double ua,vb,wc;
  double vrc[3];

  double dt =  d_params(isp,jsp).dt_verlet;
  double sigma_LJ =  d_params(isp,jsp).sigma;
  double epsilon_LJ =  d_params(isp,jsp).epsilon;
  double f11_12[3], f11_21[3], f11_22[3], f12_21[3], f12_22[3], f21_22[3];
  double f11[3], f12[3], f21[3], f22[3];
  double q11[3], q12[3], q21[3], q22[3];
  double x11s[3], x12s[3], x21s[3], x22s[3], v11s[3], v12s[3], v21s[3], v22s[3];

  //double dt_dsmc = update->dt;

  double g1, g2, s1[3], s2[3];
  double d;
  double tol = 1e-16;

  double d_11_21 ;
  double d_11_22;
  double d_12_21;
  double d_12_22;

  double err1,err2, k1, k2;

  // Setup the initial conditions
  double x0_1, x0_2,y0_1, y0_2;

  // Particle j initially stationary at (D_cutoff, b)
  x21s[0] = x22s[0] = precoln.D_cutoff;
  x21s[1] = x22s[1] = pow(rand_gen.drand(), 0.5) * precoln.bmax;
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
  double theta1 = acos( 2.0*rand_gen.drand() - 1.0);
  double theta2 = acos( 2.0*rand_gen.drand() - 1.0);

  double phi1 = rand_gen.drand()*MY_2PI;
  double phi2 = rand_gen.drand()*MY_2PI;
  
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

  double eta1 = rand_gen.drand()*MY_2PI;
  double eta2 = rand_gen.drand()*MY_2PI;

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

  for (int i=0;i< d_params(isp,jsp).timesteps;i++){
    d_11_21 = sqrt(  pow( x11s[0]-x21s[0], 2) + pow(x11s[1]-x21s[1], 2) + pow(x11s[2]-x21s[2], 2) );
    d_11_22 = sqrt(  pow( x11s[0]-x22s[0], 2) + pow(x11s[1]-x22s[1], 2) + pow(x11s[2]-x22s[2], 2) );
    d_12_21 = sqrt(  pow( x12s[0]-x21s[0], 2) + pow(x12s[1]-x21s[1], 2) + pow(x12s[2]-x21s[2], 2) );
    d_12_22 = sqrt(  pow( x12s[0]-x22s[0], 2) + pow(x12s[1]-x22s[1], 2) + pow(x12s[2]-x22s[2], 2) );

    for (int k=0;k<3;k++){
      f11_21[k]  = (( x11s[k] - x21s[k] ) / d_11_21 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_11_21,13) - pow(sigma_LJ/d_11_21,7)) ;
      f11_22[k]  = (( x11s[k] - x22s[k] ) / d_11_22 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_11_22,13) - pow(sigma_LJ/d_11_22,7)) ;
      f12_21[k]  = (( x12s[k] - x21s[k] ) / d_12_21 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_12_21,13) - pow(sigma_LJ/d_12_21,7)) ;
      f12_22[k]  = (( x12s[k] - x22s[k] ) / d_12_22 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_12_22,13) - pow(sigma_LJ/d_12_22,7)) ;

      f11[k] =  - f11_21[k] - f11_22[k];
      f12[k] =  - f12_21[k] - f12_22[k];
      f21[k] =  f12_21[k] + f11_21[k];
      f22[k] = f12_22[k] + f11_22[k];

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

    d_11_21 = sqrt(  pow( x11s[0]-x21s[0], 2) + pow(x11s[1]-x21s[1], 2) + pow(x11s[2]-x21s[2], 2) );
    d_11_22 = sqrt(  pow( x11s[0]-x22s[0], 2) + pow(x11s[1]-x22s[1], 2) + pow(x11s[2]-x22s[2], 2) );
    d_12_21 = sqrt(  pow( x12s[0]-x21s[0], 2) + pow(x12s[1]-x21s[1], 2) + pow(x12s[2]-x21s[2], 2) );
    d_12_22 = sqrt(  pow( x12s[0]-x22s[0], 2) + pow(x12s[1]-x22s[1], 2) + pow(x12s[2]-x22s[2], 2) );

    for (int k=0;k<3;k++){
      f11_21[k]  = (( x11s[k] - x21s[k] ) / d_11_21 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_11_21,13) - pow(sigma_LJ/d_11_21,7)) ;
      f11_22[k]  = (( x11s[k] - x22s[k] ) / d_11_22 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_11_22,13) - pow(sigma_LJ/d_11_22,7)) ;
      f12_21[k]  = (( x12s[k] - x21s[k] ) / d_12_21 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_12_21,13) - pow(sigma_LJ/d_12_21,7)) ;
      f12_22[k]  = (( x12s[k] - x22s[k] ) / d_12_22 )* (-24) * (epsilon_LJ/sigma_LJ ) * ( 2*pow(sigma_LJ/d_12_22,13) - pow(sigma_LJ/d_12_22,7)) ;

      f11[k] =  - f11_21[k] - f11_22[k];
      f12[k] =  - f12_21[k] - f12_22[k];
      f21[k] =  f12_21[k] + f11_21[k];
      f22[k] = f12_22[k] + f11_22[k]; 

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

  // double omega1[3], omega2[3];
  
  // omega1[0] = ((v11s[1]-vcm_post_1[1])* ((x11s[2] - x12s[2])/2 ) - (v11s[2]-vcm_post_1[2])* ((x11s[1] - x12s[1])/2 ) )/ (pow((x11s[0] - x12s[0])/2,2) + pow((x11s[1] - x12s[1])/2,2) + pow((x11s[2] - x12s[2])/2,2));
  // omega1[1] = ((v11s[2]-vcm_post_1[2])* ((x11s[0] - x12s[0])/2 ) - (v11s[0]-vcm_post_1[0])* ((x11s[2] - x12s[2])/2 ) )/ (pow((x11s[0] - x12s[0])/2,2) + pow((x11s[1] - x12s[1])/2,2) + pow((x11s[2] - x12s[2])/2,2));
  // omega1[2] = ((v11s[0]-vcm_post_1[0])* ((x11s[1] - x12s[1])/2 ) - (v11s[1]-vcm_post_1[1])* ((x11s[0] - x12s[0])/2 ) )/ (pow((x11s[0] - x12s[0])/2,2) + pow((x11s[1] - x12s[1])/2,2) + pow((x11s[2] - x12s[2])/2,2));
  
  // omega2[0] = ((v21s[1]-vcm_post_2[1])* ((x21s[2] - x22s[2])/2 ) - (v21s[2]-vcm_post_2[2])* ((x21s[1] - x22s[1])/2 ) )/ (pow((x21s[0] - x22s[0])/2,2) + pow((x21s[1] - x22s[1])/2,2) + pow((x21s[2] - x22s[2])/2,2));
  // omega2[1] = ((v21s[2]-vcm_post_2[2])* ((x21s[0] - x22s[0])/2 ) - (v21s[0]-vcm_post_2[0])* ((x21s[2] - x22s[2])/2 ) )/ (pow((x21s[0] - x22s[0])/2,2) + pow((x21s[1] - x22s[1])/2,2) + pow((x21s[2] - x22s[2])/2,2));
  // omega2[2] = ((v21s[0]-vcm_post_2[0])* ((x21s[1] - x22s[1])/2 ) - (v21s[1]-vcm_post_2[1])* ((x21s[0] - x22s[0])/2 ) )/ (pow((x21s[0] - x22s[0])/2,2) + pow((x21s[1] - x22s[1])/2,2) + pow((x21s[2] - x22s[2])/2,2));

  // ip->erot = 0.5 * I1 * (pow( omega1[0], 2) + pow(omega1[1], 2) + pow(omega1[2], 2)) ;
  // jp->erot = 0.5 * I2 * (pow( omega2[0], 2) + pow(omega2[1], 2) + pow(omega2[2], 2));

  double omega1 = (pow((v11s[0] - v12s[0])/2,2) + pow((v11s[1] - v12s[1])/2,2) +pow((v11s[2] - v12s[2])/2,2));
  double omega2 = (pow((v21s[0] - v22s[0])/2,2) + pow((v21s[1] - v22s[1])/2,2) +pow((v21s[2] - v22s[2])/2,2));

  ip->erot = atom_mass_i * omega1;
  jp->erot = atom_mass_j * omega2;

  postcoln.erot = ip->erot + jp->erot;

  //postcoln.etrans = atom_mass_i * (pow(vcm_post_1[0],2) +  pow(vcm_post_1[1],2) + pow(vcm_post_1[2],2)) + atom_mass_j * (pow(vcm_post_2[0],2) +  pow(vcm_post_2[1],2) + pow(vcm_post_2[2],2));
  postcoln.etrans = 0.5 *  d_params(isp,jsp).mr * (pow( vcm_post_1[0] - vcm_post_2[0], 2) +pow( vcm_post_1[1] - vcm_post_2[1], 2) +pow( vcm_post_1[2] - vcm_post_2[2], 2) );
  // New particle velocities. Requires postcoln.etrans to be set.
  // printf("%.5e\n",precoln.erot+precoln.etrans);
  // printf("%.5e\n",postcoln.erot+postcoln.etrans);

  double coschi = vcm_post_1[0] / sqrt( pow(vcm_post_1[0],2) +  pow(vcm_post_1[1],2) + pow(vcm_post_1[2],2) );

  double sinchi = sin(acos(coschi));
  double eps = rand_gen.drand() * 2*MY_PI;

  double *vi = ip->v;
  double *vj = jp->v;

  vrc[0] = vi[0]-vj[0];
  vrc[1] = vi[1]-vj[1];
  vrc[2] = vi[2]-vj[2];

  double scale = sqrt((2.0 * postcoln.etrans) / ( d_params(isp,jsp).mr * precoln.vr2));
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

/* ---------------------------------------------------------------------- */
void CollideDMSKokkos::SCATTER_MonatomicScatter(Particle::OnePart *ip,
                                          Particle::OnePart *jp,
                                          struct State &precoln, struct State &postcoln,
                                          rand_type &rand_gen) const
{   
  
  int isp = ip->ispecies;
  int jsp = jp->ispecies;
  double mass_i = d_species[isp].mass;
  double mass_j = d_species[jsp].mass;
  // Generate b

  double ua,vb,wc;
  double vrc[3];

  double dt = d_params(isp,jsp).dt_verlet;
  double sigma_LJ = d_params(isp,jsp).sigma;
  double epsilon_LJ = d_params(isp,jsp).epsilon;
  double dist, d;
  double f1[2], f2, x2s[2], x1s[2], v1s[2], v2s[2];

  // Separate this off to a separate function later returning chi?

  x1s[0]=0.;
  x1s[1]=0.;

  x2s[0] = precoln.D_cutoff;
  double b = pow(rand_gen.drand(), 0.5) * precoln.bmax; 
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

  for (int i=0;i<d_params(isp,jsp).timesteps;i++){
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

  // To here 

  double coschi = v1s[0] / sqrt( pow(v1s[0],2) +  pow(v1s[1],2) );
  double sinchi = v1s[1] / sqrt( pow(v1s[0],2) +  pow(v1s[1],2) );
  double eps = rand_gen.drand() * MY_2PI;

  double *vi = ip->v;
  double *vj = jp->v;

  vrc[0] = vi[0]-vj[0];
  vrc[1] = vi[1]-vj[1];
  vrc[2] = vi[2]-vj[2];

  postcoln.etrans = precoln.etrans;
  double scale = sqrt((2.0 * postcoln.etrans) / (d_params(isp,jsp).mr * precoln.vr2));
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

/* ####################################
  Don't touch below here
#######################################*/ 


/* ----------------------------------------------------------------------
   pack icell values for per-cell arrays into buf
   if icell is a split cell, also pack all sub cell values
   return byte count of amount packed
   if memflag, only return count, do not fill buf
   NOTE: why packing/unpacking parent cell if a split cell?
------------------------------------------------------------------------- */

int CollideDMSKokkos::pack_grid_one(int icell, char *buf_char, int memflag)
{
  double* buf = (double*) buf_char;

  Grid::ChildCell *cells = grid->cells;

  int n = 0;
  if (memflag) {
    for (int igroup = 0; igroup < ngroups; igroup++) {
      for (int jgroup = 0; jgroup < ngroups; jgroup++) {
        buf[n++] = k_vremax.h_view(icell,igroup,jgroup);
        if (remainflag)
          buf[n++] = k_remain.h_view(icell,igroup,jgroup);
      }
    }
  } else {
    n += ngroups*ngroups;
    if (remainflag)
      n += ngroups*ngroups;
  }

  if (cells[icell].nsplit > 1) {
    int isplit = cells[icell].isplit;
    int nsplit = cells[icell].nsplit;
    for (int i = 0; i < nsplit; i++) {
      int m = grid->sinfo[isplit].csubs[i];
      if (memflag) {
        for (int igroup = 0; igroup < ngroups; igroup++) {
          for (int jgroup = 0; jgroup < ngroups; jgroup++) {
            buf[n++] = k_vremax.h_view(m,igroup,jgroup);
            if (remainflag)
              buf[n++] = k_remain.h_view(m,igroup,jgroup);
          }
        }
      } else {
        n += ngroups*ngroups;
        if (remainflag)
          n += ngroups*ngroups;
      }
    }
  }

  return n*sizeof(double);
}

/* ----------------------------------------------------------------------
   unpack icell values for per-cell arrays from buf
   if icell is a split cell, also unpack all sub cell values
   return byte count of amount unpacked
------------------------------------------------------------------------- */

int CollideDMSKokkos::unpack_grid_one(int icell, char *buf_char)
{
  double* buf = (double*) buf_char;

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;

  grow_percell(1);
  int n = 0;
  for (int igroup = 0; igroup < ngroups; igroup++) {
    for (int jgroup = 0; jgroup < ngroups; jgroup++) {
      k_vremax.h_view(icell,igroup,jgroup) = buf[n++];
      if (remainflag)
        k_remain.h_view(icell,igroup,jgroup) = buf[n++];
    }
  }
  nglocal++;

  if (cells[icell].nsplit > 1) {
    int isplit = cells[icell].isplit;
    int nsplit = cells[icell].nsplit;
    grow_percell(nsplit);
    for (int i = 0; i < nsplit; i++) {
      int m = sinfo[isplit].csubs[i];
      for (int igroup = 0; igroup < ngroups; igroup++) {
        for (int jgroup = 0; jgroup < ngroups; jgroup++) {
          k_vremax.h_view(m,igroup,jgroup) = buf[n++];
          if (remainflag)
            k_remain.h_view(m,igroup,jgroup) = buf[n++];
        }
      }
    }
    nglocal += nsplit;
  }

  return n*sizeof(double);
}

/* ----------------------------------------------------------------------
   copy per-cell collision info from Icell to Jcell
   called whenever a grid cell is removed from this processor's list
   caller checks that Icell != Jcell
------------------------------------------------------------------------- */

void CollideDMSKokkos::copy_grid_one(int icell, int jcell)
{
  this->sync(Host,ALL_MASK);
  for (int igroup = 0; igroup < ngroups; igroup++) {
    for (int jgroup = 0; jgroup < ngroups; jgroup++) {
      k_vremax.h_view(jcell,igroup,jgroup) = k_vremax.h_view(icell,igroup,jgroup);
      if (remainflag)
        k_remain.h_view(jcell,igroup,jgroup) = k_remain.h_view(icell,igroup,jgroup);
    }
  }
  this->modified(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   reset final grid cell count after grid cell removals
------------------------------------------------------------------------- */

void CollideDMSKokkos::reset_grid_count(int nlocal)
{
  nglocal = nlocal;
}

/* ----------------------------------------------------------------------
   add a grid cell
   called when a grid cell is added to this processor's list
   initialize values to 0.0
------------------------------------------------------------------------- */

void CollideDMSKokkos::add_grid_one()
{
  grow_percell(1);

  this->sync(Host,ALL_MASK);
  for (int igroup = 0; igroup < ngroups; igroup++)
    for (int jgroup = 0; jgroup < ngroups; jgroup++) {
      k_vremax.h_view(nglocal,igroup,jgroup) = vremax_initial[igroup][jgroup];
      if (remainflag) k_remain.h_view(nglocal,igroup,jgroup) = 0.0;
    }
  this->modified(Host,ALL_MASK);

  nglocal++;
}

/* ----------------------------------------------------------------------
   reinitialize per-cell arrays due to grid cell adaptation
   count of owned grid cells has changed
   called from adapt_grid
------------------------------------------------------------------------- */

void CollideDMSKokkos::adapt_grid()
{
  int nglocal_old = nglocal;
  nglocal = grid->nlocal;

  // reallocate vremax and remain
  // initialize only new added locations
  // this leaves vremax/remain for non-adapted cells the same

  this->sync(Host,ALL_MASK);
  this->modified(Host,ALL_MASK); // force resize on host

  nglocalmax = nglocal;
  k_vremax.resize(nglocalmax,ngroups,ngroups);
  d_vremax = k_vremax.d_view;
  if (remainflag) {
    k_remain.resize(nglocalmax,ngroups,ngroups);
    d_remain = k_remain.d_view;
  }
  this->sync(Host,ALL_MASK);
  for (int icell = nglocal_old; icell < nglocal; icell++)
    for (int igroup = 0; igroup < ngroups; igroup++)
      for (int jgroup = 0; jgroup < ngroups; jgroup++) {
        k_vremax.h_view(icell,igroup,jgroup) = vremax_initial[igroup][jgroup];
        if (remainflag) k_remain.h_view(icell,igroup,jgroup) = 0.0;
      }

  this->modified(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   insure per-cell arrays are allocated long enough for N new cells
------------------------------------------------------------------------- */

void CollideDMSKokkos::grow_percell(int n)
{
  if (nglocal+n < nglocalmax || !ngroups) return;
  while (nglocal+n >= nglocalmax) nglocalmax += DELTAGRID;

  this->sync(Device,ALL_MASK); // force resize on device

  k_vremax.resize(nglocalmax,ngroups,ngroups);
  d_vremax = k_vremax.d_view;
  if (remainflag) {
    k_remain.resize(nglocalmax,ngroups,ngroups);
    d_remain = k_remain.d_view;
  }

  this->modified(Device,ALL_MASK); // needed for auto sync
}

/* ---------------------------------------------------------------------- */

void CollideDMSKokkos::sync(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if (sparta->kokkos->auto_sync)
      modified(Host,mask);
    if (mask & VREMAX_MASK) k_vremax.sync_device();
    if (remainflag)
      if (mask & REMAIN_MASK) k_remain.sync_device();
  } else {
    if (mask & VREMAX_MASK) k_vremax.sync_host();
    if (remainflag)
      if (mask & REMAIN_MASK) k_remain.sync_host();
  }
}

/* ---------------------------------------------------------------------- */

void CollideDMSKokkos::modified(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if (mask & VREMAX_MASK) k_vremax.modify_device();
    if (remainflag)
      if (mask & REMAIN_MASK) k_remain.modify_device();
    if (sparta->kokkos->auto_sync)
      sync(Host,mask);
  } else {
    if (mask & VREMAX_MASK) k_vremax.modify_host();
    if (remainflag)
      if (mask & REMAIN_MASK) k_remain.modify_host();
  }
}

/* ---------------------------------------------------------------------- */

void CollideDMSKokkos::backup()
{
  d_particles_backup = decltype(d_particles)(Kokkos::view_alloc("collide:particles_backup",Kokkos::WithoutInitializing),d_particles.extent(0));
  d_plist_backup = decltype(d_plist)(Kokkos::view_alloc("collide:plist_backup",Kokkos::WithoutInitializing),d_plist.extent(0),d_plist.extent(1));
  d_vremax_backup = decltype(d_vremax)(Kokkos::view_alloc("collide:vremax_backup",Kokkos::WithoutInitializing),d_vremax.extent(0),d_vremax.extent(1),d_vremax.extent(2));
  d_remain_backup = decltype(d_remain)(Kokkos::view_alloc("collide:remain_backup",Kokkos::WithoutInitializing),d_remain.extent(0),d_remain.extent(1),d_remain.extent(2));

  if (ambiflag) {
    error->all(FLERR,"Ambipolar collisions not supported with Kokkos");
    }

  Kokkos::deep_copy(d_particles_backup,d_particles);
  Kokkos::deep_copy(d_plist_backup,d_plist);
  Kokkos::deep_copy(d_vremax_backup,d_vremax);
  Kokkos::deep_copy(d_remain_backup,d_remain);

  if (ambiflag) {
    error->all(FLERR,"Ambipolar collisions not supported with Kokkos");
  }

  if (react) {
    ReactBirdKokkos* react_kk = (ReactBirdKokkos*) react;
    react_kk->backup();
  }

#ifdef SPARTA_KOKKOS_EXACT
  if (!random_backup)
    random_backup = new RanKnuth(12345 + comm->me);
  memcpy(random_backup,random,sizeof(RanKnuth));
#endif

}

/* ---------------------------------------------------------------------- */

void CollideDMSKokkos::restore()
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  Kokkos::deep_copy(particle_kk->k_particles.d_view,d_particles_backup);
  d_particles = particle_kk->k_particles.d_view;

  GridKokkos* grid_kk = (GridKokkos*) grid;
  Kokkos::deep_copy(grid_kk->d_plist,d_plist_backup);
  d_plist = grid_kk->d_plist;

  Kokkos::deep_copy(d_vremax,d_vremax_backup);
  Kokkos::deep_copy(d_remain,d_remain_backup);

  if (ambiflag) {
    error->all(FLERR,"Ambipolar collisions not supported with Kokkos");
  }

  if (react) {
    ReactBirdKokkos* react_kk = (ReactBirdKokkos*) react;
    react_kk->restore();
  }

#ifdef SPARTA_KOKKOS_EXACT
  memcpy(random,random_backup,sizeof(RanKnuth));
#endif

  //  reset counters

  if (sparta->kokkos->atomic_reduction) {
    h_nattempt_one() = 0;
    h_ncollide_one() = 0;
    h_nreact_one() = 0;
  }

  // deallocate references to reduce memory use

  d_particles_backup = decltype(d_particles_backup)();
  d_plist_backup = decltype(d_plist_backup)();
  d_vremax_backup = decltype(d_vremax_backup)();
  d_remain_backup = decltype(d_remain_backup)();

  if (ambiflag) {
    error->all(FLERR,"Ambipolar collisions not supported with Kokkos");
  }
}

