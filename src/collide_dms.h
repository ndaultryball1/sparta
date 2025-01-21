#ifdef COLLIDE_CLASS

CollideStyle(dms,CollideDMS)

#else

#ifndef SPARTA_COLLIDE_DMS_H
#define SPARTA_COLLIDE_DMS_H

#include "collide.h"
#include "particle.h"

#include "torch/torch.h"
#include "collide_nn.h"

namespace SPARTA_NS {
class CollideDMS : public Collide {
  public:
    CollideDMS(class SPARTA *, int, char **);
    virtual ~CollideDMS();
    virtual void init();

    double vremax_init(int, int);
    virtual double attempt_collision(int, int, double);
    double attempt_collision(int,int,int,double);
    virtual int test_collision(int, int,int, Particle::OnePart *, Particle::OnePart *);
    virtual void setup_collision(Particle::OnePart *, Particle::OnePart * );
    virtual int perform_collision( Particle::OnePart *&, Particle::OnePart *&,
                          Particle::OnePart *&);
    double extract(int, int, const char *);
  
  struct State {
    double vr2;
    double vr;
    double imass, jmass;
    double ave_rotdof;
    double ave_vibdof;
    double ave_dof;
    double etrans;
    double erot;
    double evib; // Necessary?
    double eexchange;
    double eint;
    double etotal;
    double ucmf;
    double vcmf;
    double wcmf;

    // Not strictly state of the collision
    double bmax;
    double D_cutoff;
  };

  struct Params { // Parameters for Lennard-Jones collision between species
    // What if we generalise to another potential? Would further subclass
    double sigma; // Length scale for the potential
    double epsilon; // Energy scale for the potential
    double A; // Power law parametrisation of b_max
    double B;
    double C;

    int timesteps; // Number of verlet timesteps
    double dt_verlet; // Verlet timestep size
    
    double mr;
    double bond_length_i;
    double bond_length_j; // Doesn't really make sense but unsure how else to specify. Don't want to change species parsing logic.
  };

  NNModel CollisionModel = NNModel(2,50,1); // Later this will have to be some array for inter-species collisions?
  
  protected:
    int typeflag;
    struct State precoln;
    struct State postcoln;

    Params** params;
    int nparams;                // # of per-species params read in

    void SCATTER_MonatomicScatter(Particle::OnePart *,
                                Particle::OnePart *);

    void SCATTER_RigidDiatomicScatter(Particle::OnePart *,
                                Particle::OnePart *);

    void SCATTER_VibDiatomicScatter(Particle::OnePart *,
                                Particle::OnePart *);

    void read_param_file(char *); // Evaluate how different these are to VSS and maybe push up to collide.cpp
    int wordparse(int, char *, char **);
    void setup_model();

  };
}

#endif
#endif
