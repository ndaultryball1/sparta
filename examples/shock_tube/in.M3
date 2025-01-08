################################################################################
# shock in quasi-1D tube with collisions
# particles reflect off global box boundaries
#
# Notes:
#  - The domain is only 1 cell high in Y
#  - The "comm/sort” option to the “global” command is used to match MPI runs.
#  - The “twopass” option is used to match Kokkos runs.
# The "comm/sort" and "twopass" options should not be used for production runs.
################################################################################

#  (1 cell high in Y)
# particles reflect off global box boundaries

seed	    	    12345 
dimension   	    2     
global              gridcut 0.0 comm/sort yes

boundary	    r r p

create_box  	    0 0.02 -0.005 0.005 -0.1 0.1 
create_grid 	    200 1 1
 
balance_grid        rcb cell 

region		    left block 0 0.01 -0.005 0.005 -0.1 0.1
region		    right block 0.01 0.02 -0.005 0.005 -0.1 0.1

species		    ar.species Ar Ar_2 
mixture		    gasARLeft Ar rho 1 temp 300
mixture		    gasN2Right Ar nrho 1.25e24 temp 1080

global		    fnum 1.0e17 temp 300 

create_particles    gasARLeft n 0 region left twopass # HP - high pressure region
create_particles    gasN2Right n 0 region right twopass # LP - low pressure region

collide             vss all ar.vss 

compute         gcomp grid all all u v w massrho nrho 
fix             gfix ave/grid all 1 50 50 c_gcomp[*] 
compute         gtcomp thermal/grid all all temp 
fix             gtfix ave/grid all 1 50 50 c_gtcomp[*] 
compute 	gpcomp thermal/grid all all press
fix 		gpfix ave/grid all 1 50 50 c_gpcomp[*]

stats		    100 
stats_style	    step cpu np nattempt ncoll nscoll nscheck

#dump                2 image all 50 image.*.ppm type type pdiam 3e-4 &
#                     surf proc 0.01 size 512 512 axes yes 0.9 0.02 particle yes &
#                     zoom 15 box yes 0.02
#dump_modify         2 pad 4

timestep 	    1e-9 
run 		    1500

