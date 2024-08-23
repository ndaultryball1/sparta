#ifdef COLLIDE_CLASS

CollideStyle(dms,CollideDMS)

#else

#ifndef SPARTA_COLLIDE_DMS_H
#define SPARTA_COLLIDE_DMS_H

#include "collide.h"
#include "particle.h"

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
    
    struct State {
      double vr2;
      double vr;
      double imass, jmass;
      double etrans;
      double erot;
      // double evib; // Necessary?
      // double eexchange;
      // double eint;
      double etotal;
      // double ucmf;
      // double vcmf;
      // double wcmf;
    };

    struct Params { // Parameters for collision between species
      // Delegate to a further subclass?
      double l_ref; // Length scale for the potential
      double mr;
    };
    
    protected:
      double **prefactor; // static portion of collision attempt frequency

      struct State precoln;
      struct State postcoln;

      Params** params;
      int nparams;                // # of per-species params read in

      void SCATTER_MonatomicScatter(Particle::OnePart *,
                                 Particle::OnePart *); // Decide how to implement this

      void read_param_file(char *); // Evaluate how different these are to VSS and maybe push up to collide.cpp
      int wordparse(int, char *, char **);

    };
}

#endif
#endif
