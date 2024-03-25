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
#include "math_extra.h"
#include "string.h"
#include "marching_cubes.h"
#include "grid.h"
#include "surf.h"
#include "irregular.h"
#include "lookup_table.h"
#include "geometry.h"
#include "my_page.h"
#include "memory.h"
#include "error.h"

// DEBUG
#include "update.h"

using namespace SPARTA_NS;

// prototype for non-class function

int compare_indices(const void *, const void *);

enum{UNKNOWN,OUTSIDE,INSIDE,OVERLAP};           // several files
enum{NCHILD,NPARENT,NUNKNOWN,NPBCHILD,NPBPARENT,NPBUNKNOWN,NBOUND};  // Grid

#define DELTA 128
#define BIG 1.0e20
#define EPSILON 1.0e-16

/* ---------------------------------------------------------------------- */

MarchingCubes::MarchingCubes(SPARTA *sparta, int ggroup_caller,
                             double thresh_caller) :
  Pointers(sparta)
{
  MPI_Comm_rank(world,&me);

  ggroup = ggroup_caller;
  thresh = thresh_caller;
}

/* ----------------------------------------------------------------------
   create 2d implicit surfs from grid point values
   follows https://en.wikipedia.org/wiki/Marching_squares
   see 2 sections: Basic algorithm and Disambiguation of saddle points
     treating open circles as flow volume, solid circles as material
     NOTE: Wiki page numbers points counter-clockwise
           SPARTA numbers them in x, then in y
           so bit2 and bit3 are swapped below
           this gives case #s here consistent with Wiki page
   process each grid cell independently
   4 corner points open/solid -> 2^4 = 16 cases
   cases infer 0,1,2 line segments in each grid cell
   order 2 points in each line segment to give normal into flow volume
   treat two saddle point cases (my 9,6) (Wiki 5,10)
     based on ave value at cell center
------------------------------------------------------------------------- */

void MarchingCubes::invoke(double **cvalues, int *svalues, int **mcflags)
{
  int i,ipt,isurf,nsurf,icase,which;
  surfint surfID;
  surfint *ptr;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  MyPage<surfint> *csurfs = grid->csurfs;
  int nglocal = grid->nlocal;
  int groupbit = grid->bitmask[ggroup];

  bigint maxsurfID = 0;
  if (sizeof(surfint) == 4) maxsurfID = MAXSMALLINT;
  if (sizeof(surfint) == 8) maxsurfID = MAXBIGINT;

  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;
    lo = cells[icell].lo;
    hi = cells[icell].hi;

    // nsurf = # of tris in cell
    // cvalues[8] = 8 corner point values, each is 0 to 255 inclusive
    // thresh = value between 0 and 255 to threshhold on
    // lo[3] = lower left corner pt of grid cell
    // hi[3] = upper right corner pt of grid cell
    // pt = list of 3*nsurf points that are the corner pts of each tri

    // cvalues in SPARTA are ordered
    // bottom-lower-left, bottom-lower-right,
    // bottom-upper-left, bottom-upper-right
    // top-lower-left, top-lower-right, top-upper-left, top-upper-right
    // Vzyx encodes this as 0/1 in each dim

    // ordering in cvalues different from loop up table
    // manually change for consistency

    v[0] = cvalues[icell][0];
    v[1] = cvalues[icell][1];
    v[2] = cvalues[icell][3];
    v[3] = cvalues[icell][2];
    v[4] = cvalues[icell][4];
    v[5] = cvalues[icell][5];
    v[6] = cvalues[icell][7];
    v[7] = cvalues[icell][6];

    // temporary viso values

    for (i = 0; i < 8; i++) viso[i] = v[i] - thresh;

    bit0 = v[0] <= thresh ? 0 : 1;
    bit1 = v[1] <= thresh ? 0 : 1;
    bit2 = v[2] <= thresh ? 0 : 1;
    bit3 = v[3] <= thresh ? 0 : 1;
    bit4 = v[4] <= thresh ? 0 : 1;
    bit5 = v[5] <= thresh ? 0 : 1;
    bit6 = v[6] <= thresh ? 0 : 1;
    bit7 = v[7] <= thresh ? 0 : 1;

    which = (bit7 << 7) + (bit6 << 6) + (bit5 << 5) + (bit4 << 4) +
      (bit3 << 3) + (bit2 << 2) + (bit1 << 1) + bit0;

    // icase = case of the active cube in [0..15]

    icase = cases[which][0];
    config = cases[which][1];
    subconfig = 0;

    switch (icase) {
    case  0:
      nsurf = 0;
      break;

    case  1:
      nsurf = add_triangle(tiling1[config], 1);
      break;

    case  2:
      nsurf = add_triangle(tiling2[config], 2);
      break;

    case  3:
      if (test_face(test3[config]))
        nsurf = add_triangle(tiling3_2[config], 4); // 3.2
      else
        nsurf = add_triangle(tiling3_1[config], 2); // 3.1
      break;

    case  4:
      if (modified_test_interior(test4[config],icase))
        nsurf = add_triangle(tiling4_1[config], 2); // 4.1.1
      else
        nsurf = add_triangle(tiling4_2[config], 6); // 4.1.2
      break;

    case  5:
      nsurf = add_triangle(tiling5[config], 3);
      break;

    case  6:
      if (test_face(test6[config][0]))
        nsurf = add_triangle(tiling6_2[config], 5); // 6.2
      else {
        if (modified_test_interior(test6[config][1],icase))
          nsurf = add_triangle(tiling6_1_1[config], 3); // 6.1.1
        else {
          nsurf = add_triangle(tiling6_1_2[config], 9); // 6.1.2
        }
      }
      break;

    case  7:
      if (test_face(test7[config][0])) subconfig +=  1;
      if (test_face(test7[config][1])) subconfig +=  2;
      if (test_face(test7[config][2])) subconfig +=  4;
      switch (subconfig) {
      case 0:
        nsurf = add_triangle(tiling7_1[config], 3); break;
      case 1:
        nsurf = add_triangle(tiling7_2[config][0], 5); break;
      case 2:
        nsurf = add_triangle(tiling7_2[config][1], 5); break;
      case 3:
        nsurf = add_triangle(tiling7_3[config][0], 9); break;
      case 4:
        nsurf = add_triangle(tiling7_2[config][2], 5); break;
      case 5:
        nsurf = add_triangle(tiling7_3[config][1], 9); break;
      case 6:
        nsurf = add_triangle(tiling7_3[config][2], 9); break;
      case 7:
        if (test_interior(test7[config][3],icase))
          nsurf = add_triangle(tiling7_4_2[config], 9);
        else
          nsurf = add_triangle(tiling7_4_1[config], 5);
        break;
      };
      break;

    case  8:
      nsurf = add_triangle(tiling8[config], 2);
      break;

    case  9:
      nsurf = add_triangle(tiling9[config], 4);
      break;

    case 10:
      if (test_face(test10[config][0])) {
        if (test_face(test10[config][1]))
          nsurf = add_triangle(tiling10_1_1_[config], 4); // 10.1.1
        else {
          nsurf = add_triangle(tiling10_2[config], 8); // 10.2
        }
      } else {
        if (test_face(test10[config][1])) {
          nsurf = add_triangle(tiling10_2_[config], 8); // 10.2
        } else {
          if (test_interior(test10[config][2],icase))
            nsurf = add_triangle(tiling10_1_1[config], 4); // 10.1.1
          else
            nsurf = add_triangle(tiling10_1_2[config], 8); // 10.1.2
        }
      }
      break;

    case 11:
      nsurf = add_triangle(tiling11[config], 4);
      break;

    case 12:
      if (test_face(test12[config][0])) {
        if (test_face(test12[config][1]))
          nsurf = add_triangle(tiling12_1_1_[config], 4); // 12.1.1
        else {
          nsurf = add_triangle(tiling12_2[config], 8); // 12.2
        }
      } else {
        if (test_face(test12[config][1])) {
          nsurf = add_triangle(tiling12_2_[config], 8); // 12.2
        } else {
          if (test_interior(test12[config][2],icase))
            nsurf = add_triangle(tiling12_1_1[config], 4); // 12.1.1
          else
            nsurf = add_triangle(tiling12_1_2[config], 8); // 12.1.2
        }
      }
      break;

    case 13:
      if (test_face(test13[config][0])) subconfig +=  1;
      if (test_face(test13[config][1])) subconfig +=  2;
      if (test_face(test13[config][2])) subconfig +=  4;
      if (test_face(test13[config][3])) subconfig +=  8;
      if (test_face(test13[config][4])) subconfig += 16;
      if (test_face(test13[config][5])) subconfig += 32;

      switch (subconfig13[subconfig]) {
      case 0:/* 13.1 */
        nsurf = add_triangle(tiling13_1[config], 4); break;

      case 1:/* 13.2 */
        nsurf = add_triangle(tiling13_2[config][0], 6); break;
      case 2:/* 13.2 */
        nsurf = add_triangle(tiling13_2[config][1], 6); break;
      case 3:/* 13.2 */
        nsurf = add_triangle(tiling13_2[config][2], 6); break;
      case 4:/* 13.2 */
        nsurf = add_triangle(tiling13_2[config][3], 6); break;
      case 5:/* 13.2 */
        nsurf = add_triangle(tiling13_2[config][4], 6); break;
      case 6:/* 13.2 */
        nsurf = add_triangle(tiling13_2[config][5], 6); break;

      case 7:/* 13.3 */
        nsurf = add_triangle(tiling13_3[config][0], 10); break;
      case 8:/* 13.3 */
        nsurf = add_triangle(tiling13_3[config][1], 10); break;
      case 9:/* 13.3 */
        nsurf = add_triangle(tiling13_3[config][2], 10); break;
      case 10:/* 13.3 */
        nsurf = add_triangle(tiling13_3[config][3], 10); break;
      case 11:/* 13.3 */
        nsurf = add_triangle(tiling13_3[config][4], 10); break;
      case 12:/* 13.3 */
        nsurf = add_triangle(tiling13_3[config][5], 10); break;
      case 13:/* 13.3 */
        nsurf = add_triangle(tiling13_3[config][6], 10); break;
      case 14:/* 13.3 */
        nsurf = add_triangle(tiling13_3[config][7], 10); break;
      case 15:/* 13.3 */
        nsurf = add_triangle(tiling13_3[config][8], 10); break;
      case 16:/* 13.3 */
        nsurf = add_triangle(tiling13_3[config][9], 10); break;
      case 17:/* 13.3 */
        nsurf = add_triangle(tiling13_3[config][10], 10); break;
      case 18:/* 13.3 */
        nsurf = add_triangle(tiling13_3[config][11], 10); break;

      case 19:/* 13.4 */
        nsurf = add_triangle(tiling13_4[config][0], 12); break;
      case 20:/* 13.4 */
        nsurf = add_triangle(tiling13_4[config][1], 12); break;
      case 21:/* 13.4 */
        nsurf = add_triangle(tiling13_4[config][2], 12); break;
      case 22:/* 13.4 */
        nsurf = add_triangle(tiling13_4[config][3], 12); break;

      case 23:/* 13.5 */
        subconfig = 0;
        if (interior_test_case13())
          nsurf = add_triangle(tiling13_5_1[config][0], 6);
        else
          nsurf = add_triangle(tiling13_5_2[config][0], 10);
        break;

      case 24:/* 13.5 */
        subconfig = 1;
        if (interior_test_case13())
          nsurf = add_triangle(tiling13_5_1[config][1], 6);
        else
          nsurf = add_triangle(tiling13_5_2[config][1], 10);
        break;

      case 25:/* 13.5 */
        subconfig = 2;
        if (interior_test_case13())
          nsurf = add_triangle(tiling13_5_1[config][2], 6);
        else
          nsurf = add_triangle(tiling13_5_2[config][2], 10);
        break;

      case 26:/* 13.5 */
        subconfig = 3;
        if (interior_test_case13())
          nsurf = add_triangle(tiling13_5_1[config][3], 6);
        else
          nsurf = add_triangle(tiling13_5_2[config][3], 10);
        break;

      case 27:/* 13.3 */
        nsurf = add_triangle(tiling13_3_[config][0], 10); break;
      case 28:/* 13.3 */
        nsurf = add_triangle(tiling13_3_[config][1], 10); break;
      case 29:/* 13.3 */
        nsurf = add_triangle(tiling13_3_[config][2], 10); break;
      case 30:/* 13.3 */
        nsurf = add_triangle(tiling13_3_[config][3], 10); break;
      case 31:/* 13.3 */
        nsurf = add_triangle(tiling13_3_[config][4], 10); break;
      case 32:/* 13.3 */
        nsurf = add_triangle(tiling13_3_[config][5], 10); break;
      case 33:/* 13.3 */
        nsurf = add_triangle(tiling13_3_[config][6], 10); break;
      case 34:/* 13.3 */
        nsurf = add_triangle(tiling13_3_[config][7], 10); break;
      case 35:/* 13.3 */
        nsurf = add_triangle(tiling13_3_[config][8], 10); break;
      case 36:/* 13.3 */
        nsurf = add_triangle(tiling13_3_[config][9], 10); break;
      case 37:/* 13.3 */
        nsurf = add_triangle(tiling13_3_[config][10], 10); break;
      case 38:/* 13.3 */
        nsurf = add_triangle(tiling13_3_[config][11], 10); break;

      case 39:/* 13.2 */
        nsurf = add_triangle(tiling13_2_[config][0], 6); break;
      case 40:/* 13.2 */
        nsurf = add_triangle(tiling13_2_[config][1], 6); break;
      case 41:/* 13.2 */
        nsurf = add_triangle(tiling13_2_[config][2], 6); break;
      case 42:/* 13.2 */
        nsurf = add_triangle(tiling13_2_[config][3], 6); break;
      case 43:/* 13.2 */
        nsurf = add_triangle(tiling13_2_[config][4], 6); break;
      case 44:/* 13.2 */
        nsurf = add_triangle(tiling13_2_[config][5], 6); break;

      case 45:/* 13.1 */
        nsurf = add_triangle(tiling13_1_[config], 4); break;

      default:
        print_cube();
        error->one(FLERR,"Marching cubes - impossible case 13");
      }
      break;

    case 14:
      nsurf = add_triangle(tiling14[config], 4);
      break;
    };

    // store 4 MC labels for FixAblate caller

    mcflags[icell][0] = icase;
    mcflags[icell][1] = config;
    mcflags[icell][2] = subconfig;
    mcflags[icell][3] = nsurf;

    // populate Grid and Surf data structs
    // points will be duplicated, not unique
    // surf ID = cell ID for all surfs in cell
    // check if uint cell ID overflows int surf ID

    if (nsurf) {
      if (cells[icell].id > maxsurfID)
        error->one(FLERR,"Grid cell ID overflows implicit surf ID");
      surfID = cells[icell].id;
    }

    ptr = csurfs->get(nsurf);

    ipt = 0;
    for (i = 0; i < nsurf; i++) {
      if (svalues) surf->add_tri(surfID,svalues[icell],
                                 pt[ipt+2],pt[ipt+1],pt[ipt]);
      else surf->add_tri(surfID,1,pt[ipt+2],pt[ipt+1],pt[ipt]);
      ipt += 3;
      isurf = surf->nlocal - 1;
      ptr[i] = isurf;
    }

    cells[icell].nsurf = nsurf;
    if (nsurf) {
      cells[icell].csurfs = ptr;
      cinfo[icell].type = OVERLAP;
    }
  }
}

/* ----------------------------------------------------------------------
   Same as above but uses inner values. Also if there are ambiguities,
   the corner values corresponding to the intersections are first found
   then the ambiguity tests are performed
------------------------------------------------------------------------- */

void MarchingCubes::invoke(double ***cvalues, int *svalues, int **mcflags)
{
  int i,j,ipt,isurf,nsurf,icase,which;
  surfint surfID;
  surfint *ptr;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  MyPage<surfint> *csurfs = grid->csurfs;
  int nglocal = grid->nlocal;
  int groupbit = grid->bitmask[ggroup];

  bigint maxsurfID = 0;
  if (sizeof(surfint) == 4) maxsurfID = MAXSMALLINT;
  if (sizeof(surfint) == 8) maxsurfID = MAXBIGINT;

  for (int icell = 0; icell < nglocal; icell++) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit <= 0) continue;
    lo = cells[icell].lo;
    hi = cells[icell].hi;

    // nsurf = # of tris in cell
    // cvalues[8] = 8 corner point values, each is 0 to 255 inclusive
    // thresh = value between 0 and 255 to threshhold on
    // lo[3] = lower left corner pt of grid cell
    // hi[3] = upper right corner pt of grid cell
    // pt = list of 3*nsurf points that are the corner pts of each tri

    // cvalues are ordered
    // bottom-lower-left, bottom-lower-right,
    // bottom-upper-left, bottom-upper-right
    // top-lower-left, top-lower-right, top-upper-left, top-upper-right
    // Vzyx encodes this as 0/1 in each dim

    // temporarily store all inner values

    for (i = 0; i < 8; i++)
      for (j = 0; j < 6; j++)
        inval[i][j] = cvalues[icell][i][j];

    // use averages for now

    for (i = 0; i < 8; i++) v[i] = 0.0;

    // ordering in cvalues different from loop up table
    // manually change for consistency

    for (j = 0; j < 6; j++) {
      v[0] += inval[0][j];
      v[1] += inval[1][j];
      v[2] += inval[3][j];
      v[3] += inval[2][j];
      v[4] += inval[4][j];
      v[5] += inval[5][j];
      v[6] += inval[7][j];
      v[7] += inval[6][j];
    }      

    for (i = 0; i < 8; i++) v[i] /= 6; 

    // temporary viso values

    for (i = 0; i < 8; i++) viso[i] = v[i] - thresh;

    // intersection of surfaces on all cell edges

    i0  = interpolate(inval[0][1],inval[1][0],lo[0],hi[0]);
    i1  = interpolate(inval[1][3],inval[3][2],lo[1],hi[1]);
    i2  = interpolate(inval[2][1],inval[3][0],lo[0],hi[0]);
    i3  = interpolate(inval[0][3],inval[2][2],lo[1],hi[1]);

    i4  = interpolate(inval[4][1],inval[5][0],lo[0],hi[0]);
    i5  = interpolate(inval[5][3],inval[7][2],lo[1],hi[1]);
    i6  = interpolate(inval[6][1],inval[7][0],lo[0],hi[0]);
    i7  = interpolate(inval[4][3],inval[6][2],lo[1],hi[1]);

    i8  = interpolate(inval[0][5],inval[4][4],lo[2],hi[2]);
    i9  = interpolate(inval[1][5],inval[5][4],lo[2],hi[2]);
    i10 = interpolate(inval[3][5],inval[7][4],lo[2],hi[2]);
    i11 = interpolate(inval[2][5],inval[6][4],lo[2],hi[2]);

    // intersection on unit cube

    i0u  = interpolate(inval[0][1],inval[1][0],0,1);
    i1u  = interpolate(inval[1][3],inval[3][2],0,1);
    i2u  = interpolate(inval[2][1],inval[3][0],0,1);
    i3u  = interpolate(inval[0][3],inval[2][2],0,1);

    i4u  = interpolate(inval[4][1],inval[5][0],0,1);
    i5u  = interpolate(inval[5][3],inval[7][2],0,1);
    i6u  = interpolate(inval[6][1],inval[7][0],0,1);
    i7u  = interpolate(inval[4][3],inval[6][2],0,1);

    i8u  = interpolate(inval[0][5],inval[4][4],0,1);
    i9u  = interpolate(inval[1][5],inval[5][4],0,1);
    i10u = interpolate(inval[3][5],inval[7][4],0,1);
    i11u = interpolate(inval[2][5],inval[6][4],0,1);

    // make bits 2, 3, 6 and 7 consistent with Lewiner paper (see NOTE above)

    bit0 = v[0] <= thresh ? 0 : 1;
    bit1 = v[1] <= thresh ? 0 : 1;
    bit2 = v[2] <= thresh ? 0 : 1;
    bit3 = v[3] <= thresh ? 0 : 1;
    bit4 = v[4] <= thresh ? 0 : 1;
    bit5 = v[5] <= thresh ? 0 : 1;
    bit6 = v[6] <= thresh ? 0 : 1;
    bit7 = v[7] <= thresh ? 0 : 1;

    which = (bit7 << 7) + (bit6 << 6) + (bit5 << 5) + (bit4 << 4) +
      (bit3 << 3) + (bit2 << 2) + (bit1 << 1) + bit0;

    // icase = case of the active cube in [0..15]

    icase = cases[which][0];
    config = cases[which][1];
    subconfig = 0;

    switch (icase) {
    case  0:
      nsurf = 0;
      break;

    case  1:
      nsurf = add_triangle_inner(tiling1[config], 1);
      break;

    case  2:
      nsurf = add_triangle_inner(tiling2[config], 2);
      break;

    case  3:
      if (test_face_inner(test3[config]))
        nsurf = add_triangle_inner(tiling3_2[config], 4); // 3.2
      else
        nsurf = add_triangle_inner(tiling3_1[config], 2); // 3.1
      break;

    case  4:
      if (modified_test_interior(test4[config],icase))
        nsurf = add_triangle_inner(tiling4_1[config], 2); // 4.1.1
      else
        nsurf = add_triangle_inner(tiling4_2[config], 6); // 4.1.2
      break;

    case  5:
      nsurf = add_triangle_inner(tiling5[config], 3);
      break;

    case  6:
      if (test_face_inner(test6[config][0]))
        nsurf = add_triangle_inner(tiling6_2[config], 5); // 6.2
      else {
        if (modified_test_interior(test6[config][1],icase))
          nsurf = add_triangle_inner(tiling6_1_1[config], 3); // 6.1.1
        else {
          nsurf = add_triangle_inner(tiling6_1_2[config], 9); // 6.1.2
        }
      }
      break;

    case  7:
      if (test_face_inner(test7[config][0])) subconfig +=  1;
      if (test_face_inner(test7[config][1])) subconfig +=  2;
      if (test_face_inner(test7[config][2])) subconfig +=  4;
      switch (subconfig) {
      case 0:
        nsurf = add_triangle_inner(tiling7_1[config], 3); break;
      case 1:
        nsurf = add_triangle_inner(tiling7_2[config][0], 5); break;
      case 2:
        nsurf = add_triangle_inner(tiling7_2[config][1], 5); break;
      case 3:
        nsurf = add_triangle_inner(tiling7_3[config][0], 9); break;
      case 4:
        nsurf = add_triangle_inner(tiling7_2[config][2], 5); break;
      case 5:
        nsurf = add_triangle_inner(tiling7_3[config][1], 9); break;
      case 6:
        nsurf = add_triangle_inner(tiling7_3[config][2], 9); break;
      case 7:
        if (test_interior(test7[config][3],icase))
          nsurf = add_triangle_inner(tiling7_4_2[config], 9);
        else
          nsurf = add_triangle_inner(tiling7_4_1[config], 5);
        break;
      };
      break;

    case  8:
      nsurf = add_triangle_inner(tiling8[config], 2);
      break;

    case  9:
      nsurf = add_triangle_inner(tiling9[config], 4);
      break;

    case 10:
      if (test_face_inner(test10[config][0])) {
        if (test_face_inner(test10[config][1]))
          nsurf = add_triangle_inner(tiling10_1_1_[config], 4); // 10.1.1
        else {
          nsurf = add_triangle_inner(tiling10_2[config], 8); // 10.2
        }
      } else {
        if (test_face_inner(test10[config][1])) {
          nsurf = add_triangle_inner(tiling10_2_[config], 8); // 10.2
        } else {
          if (test_interior(test10[config][2],icase))
            nsurf = add_triangle_inner(tiling10_1_1[config], 4); // 10.1.1
          else
            nsurf = add_triangle_inner(tiling10_1_2[config], 8); // 10.1.2
        }
      }
      break;

    case 11:
      nsurf = add_triangle_inner(tiling11[config], 4);
      break;

    case 12:
      if (test_face_inner(test12[config][0])) {
        if (test_face_inner(test12[config][1]))
          nsurf = add_triangle_inner(tiling12_1_1_[config], 4); // 12.1.1
        else {
          nsurf = add_triangle_inner(tiling12_2[config], 8); // 12.2
        }
      } else {
        if (test_face_inner(test12[config][1])) {
          nsurf = add_triangle_inner(tiling12_2_[config], 8); // 12.2
        } else {
          if (test_interior(test12[config][2],icase))
            nsurf = add_triangle_inner(tiling12_1_1[config], 4); // 12.1.1
          else
            nsurf = add_triangle_inner(tiling12_1_2[config], 8); // 12.1.2
        }
      }
      break;

    case 13:
      if (test_face_inner(test13[config][0])) subconfig +=  1;
      if (test_face_inner(test13[config][1])) subconfig +=  2;
      if (test_face_inner(test13[config][2])) subconfig +=  4;
      if (test_face_inner(test13[config][3])) subconfig +=  8;
      if (test_face_inner(test13[config][4])) subconfig += 16;
      if (test_face_inner(test13[config][5])) subconfig += 32;

      switch (subconfig13[subconfig]) {
      case 0:/* 13.1 */
        nsurf = add_triangle_inner(tiling13_1[config], 4); break;

      case 1:/* 13.2 */
        nsurf = add_triangle_inner(tiling13_2[config][0], 6); break;
      case 2:/* 13.2 */
        nsurf = add_triangle_inner(tiling13_2[config][1], 6); break;
      case 3:/* 13.2 */
        nsurf = add_triangle_inner(tiling13_2[config][2], 6); break;
      case 4:/* 13.2 */
        nsurf = add_triangle_inner(tiling13_2[config][3], 6); break;
      case 5:/* 13.2 */
        nsurf = add_triangle_inner(tiling13_2[config][4], 6); break;
      case 6:/* 13.2 */
        nsurf = add_triangle_inner(tiling13_2[config][5], 6); break;

      case 7:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3[config][0], 10); break;
      case 8:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3[config][1], 10); break;
      case 9:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3[config][2], 10); break;
      case 10:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3[config][3], 10); break;
      case 11:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3[config][4], 10); break;
      case 12:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3[config][5], 10); break;
      case 13:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3[config][6], 10); break;
      case 14:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3[config][7], 10); break;
      case 15:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3[config][8], 10); break;
      case 16:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3[config][9], 10); break;
      case 17:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3[config][10], 10); break;
      case 18:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3[config][11], 10); break;

      case 19:/* 13.4 */
        nsurf = add_triangle_inner(tiling13_4[config][0], 12); break;
      case 20:/* 13.4 */
        nsurf = add_triangle_inner(tiling13_4[config][1], 12); break;
      case 21:/* 13.4 */
        nsurf = add_triangle_inner(tiling13_4[config][2], 12); break;
      case 22:/* 13.4 */
        nsurf = add_triangle_inner(tiling13_4[config][3], 12); break;

      case 23:/* 13.5 */
        subconfig = 0;
        if (interior_test_case13())
          nsurf = add_triangle_inner(tiling13_5_1[config][0], 6);
        else
          nsurf = add_triangle_inner(tiling13_5_2[config][0], 10);
        break;

      case 24:/* 13.5 */
        subconfig = 1;
        if (interior_test_case13())
          nsurf = add_triangle_inner(tiling13_5_1[config][1], 6);
        else
          nsurf = add_triangle_inner(tiling13_5_2[config][1], 10);
        break;

      case 25:/* 13.5 */
        subconfig = 2;
        if (interior_test_case13())
          nsurf = add_triangle_inner(tiling13_5_1[config][2], 6);
        else
          nsurf = add_triangle_inner(tiling13_5_2[config][2], 10);
        break;

      case 26:/* 13.5 */
        subconfig = 3;
        if (interior_test_case13())
          nsurf = add_triangle_inner(tiling13_5_1[config][3], 6);
        else
          nsurf = add_triangle_inner(tiling13_5_2[config][3], 10);
        break;

      case 27:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3_[config][0], 10); break;
      case 28:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3_[config][1], 10); break;
      case 29:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3_[config][2], 10); break;
      case 30:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3_[config][3], 10); break;
      case 31:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3_[config][4], 10); break;
      case 32:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3_[config][5], 10); break;
      case 33:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3_[config][6], 10); break;
      case 34:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3_[config][7], 10); break;
      case 35:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3_[config][8], 10); break;
      case 36:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3_[config][9], 10); break;
      case 37:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3_[config][10], 10); break;
      case 38:/* 13.3 */
        nsurf = add_triangle_inner(tiling13_3_[config][11], 10); break;

      case 39:/* 13.2 */
        nsurf = add_triangle_inner(tiling13_2_[config][0], 6); break;
      case 40:/* 13.2 */
        nsurf = add_triangle_inner(tiling13_2_[config][1], 6); break;
      case 41:/* 13.2 */
        nsurf = add_triangle_inner(tiling13_2_[config][2], 6); break;
      case 42:/* 13.2 */
        nsurf = add_triangle_inner(tiling13_2_[config][3], 6); break;
      case 43:/* 13.2 */
        nsurf = add_triangle_inner(tiling13_2_[config][4], 6); break;
      case 44:/* 13.2 */
        nsurf = add_triangle_inner(tiling13_2_[config][5], 6); break;

      case 45:/* 13.1 */
        nsurf = add_triangle_inner(tiling13_1_[config], 4); break;

      default:
        print_cube();
        error->one(FLERR,"Marching cubes - impossible case 13");
      }
      break;

    case 14:
      nsurf = add_triangle_inner(tiling14[config], 4);
      break;
    };

    // store 4 MC labels for FixAblate caller

    mcflags[icell][0] = icase;
    mcflags[icell][1] = config;
    mcflags[icell][2] = subconfig;
    mcflags[icell][3] = nsurf;

    // populate Grid and Surf data structs
    // points will be duplicated, not unique
    // surf ID = cell ID for all surfs in cell
    // check if uint cell ID overflows int surf ID

    if (nsurf) {
      if (cells[icell].id > maxsurfID)
        error->one(FLERR,"Grid cell ID overflows implicit surf ID");
      surfID = cells[icell].id;
    }

    ptr = csurfs->get(nsurf);

    ipt = 0;
    for (i = 0; i < nsurf; i++) {
      if (svalues) surf->add_tri(surfID,svalues[icell],
                                 pt[ipt+2],pt[ipt+1],pt[ipt]);
      else surf->add_tri(surfID,1,pt[ipt+2],pt[ipt+1],pt[ipt]);
      ipt += 3;
      isurf = surf->nlocal - 1;
      ptr[i] = isurf;
    }

    cells[icell].nsurf = nsurf;
    if (nsurf) {
      cells[icell].csurfs = ptr;
      cinfo[icell].type = OVERLAP;
    }
  }
}


/* ----------------------------------------------------------------------
   interpolate function used by both marching squares and cubes
   lo/hi = coordinates of end points of edge of square
   v0/v1 = values at lo/hi end points
   value = interpolated coordinate for thresh value
------------------------------------------------------------------------- */

double MarchingCubes::interpolate(double v0, double v1, double lo, double hi)
{
  double value = lo + (hi-lo)*(thresh-v0)/(v1-v0);
  value = MAX(value,lo);
  value = MIN(value,hi);
  return value;
}

/* ----------------------------------------------------------------------
   clean up issues that marching cubes occasionally generates
     that cause problems for SPARTA
   what MC does:
     may generate 0 or 2 triangles on the face of a cell
     the cell sharing the face may also generate 0 or 2 triangles
     the normals for the 2 triangles may be into or out of the owning cell
   what SPARTA needs:
     let cell1 and cell2 be two cells that share a face
     if cell1 has 2 tris on face and cell2 has none:
       if norm is into cell1: keep them in cell1
       if norm is into cell2: assign both tris to cell2
     if both cell1 and cell2 have 2 tris on face: delete all 4 tris
   algorithm to do this:
     loop over all my cells with implicit tris:
       count how many surfs on each face
     loop over all my cells with implicit tris:
       loop over faces with 2 tris:
         if I own adjoining cell:
           check its tally on shared face
           reassign and/or delete triangles as necessary
         if I do not own adjoining cell:
           add 2 tris to send list for this proc
     irregular comm of send list to nearby procs (share faces of my cells)
     each proc loops over its recv list:
       if my cell face has 2 tris: delete them
       if my cell face has 0 tris: skip or add 2 tris depending on norm
 ------------------------------------------------------------------------- */

void MarchingCubes::cleanup()
{
  int i,j,k,m,icell,iface,nsurf,idim,nflag,inwardnorm;
  int ntri_other,othercell,otherface,otherproc,otherlocal,othernsurf;
  surfint *oldcsurfs;
  surfint *ptr;
  double *lo,*hi;
  double *norm;

  Surf::Tri *tris = surf->tris;
  Grid::ChildCell *cells = grid->cells;
  MyPage<surfint> *csurfs = grid->csurfs;
  int nglocal = grid->nlocal;

  // count # of tris on each face of every cell I own

  int **nfacetri;
  int ***facetris;
  memory->create(nfacetri,nglocal,6,"readisurf:nfacetri");
  memory->create(facetris,nglocal,6,2,"readisurf:facetris");

  for (icell = 0; icell < nglocal; icell++) {
    nfacetri[icell][0] = nfacetri[icell][1] = nfacetri[icell][2] =
      nfacetri[icell][3] = nfacetri[icell][4] = nfacetri[icell][5] = 0;

    if (cells[icell].nsplit <= 0) continue;
    nsurf = cells[icell].nsurf;
    if (nsurf == 0) continue;

    lo = cells[icell].lo;
    hi = cells[icell].hi;

    for (j = 0; j < nsurf; j++) {
      m = cells[icell].csurfs[j];
      iface = Geometry::tri_on_hex_face(tris[m].p1,tris[m].p2,tris[m].p3,lo,hi);
      if (iface < 0) continue;
      if (nfacetri[icell][iface] < 2)
        facetris[icell][iface][nfacetri[icell][iface]] = m;
      nfacetri[icell][iface]++;
    }
  }

  // check that every face has 0 or 2 tris

  int flag = 0;
  for (icell = 0; icell < nglocal; icell++)
    for (iface = 0; iface < 6; iface++)
      if (nfacetri[icell][iface] != 0 && nfacetri[icell][iface] != 2)
        flag++;

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall)
    error->all(FLERR,"Some cell faces do not have zero or 2 triangles");

  // loop over all cell faces
  // check tri count for that face for both adjoining cells

  int *proclist = NULL;
  SendDatum *bufsend = NULL;
  int nsend = 0;
  int maxsend = 0;

  int *dellist = NULL;
  int ndelete = 0;
  int maxdelete = 0;

  // DEBUG
  //int ntotal = 0;
  //int nadd = 0;
  //int ndel = 0;

  for (icell = 0; icell < nglocal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    nsurf = cells[icell].nsurf;
    if (nsurf == 0) continue;

    for (iface = 0; iface < 6; iface++) {
      if (nfacetri[icell][iface] != 2) continue;
      //ntotal += 2;

      // other cell/face/proc = info for matching face in adjacent cell

      nflag = grid->neigh_decode(cells[icell].nmask,iface);
      if (nflag != NCHILD && nflag != NPBCHILD)
        error->one(FLERR,"Invalid neighbor cell in cleanup_MC()");

      norm = tris[facetris[icell][iface][0]].norm;
      idim = iface/2;
      if (iface % 2 && norm[idim] < 0.0) inwardnorm = 1;
      else if (iface % 2 == 0 && norm[idim] > 0.0) inwardnorm = 1;
      else inwardnorm = 0;
      if (iface % 2) otherface = iface-1;
      else otherface = iface+1;
      othercell = (int) cells[icell].neigh[iface];
      otherproc = cells[othercell].proc;
      otherlocal = cells[othercell].ilocal;

      // if I own the adjacent cell, make decision about shared tris
      // if both cells have 2 tris on face, delete all of them
      // otherwise cell that matches inward normal is assigned the 2 tris

      if (otherproc == me) {
        ntri_other = nfacetri[othercell][otherface];

        // icell keeps the 2 tris

        if (ntri_other == 0 && inwardnorm) continue;

        // add 2 tris to othercell
        // reset tri IDs to new owning cell

        if (ntri_other == 0) {
          othernsurf = cells[othercell].nsurf;
          oldcsurfs = cells[othercell].csurfs;
          ptr = csurfs->get(othernsurf+2);
          for (k = 0; k < othernsurf; k++)
            ptr[k] = oldcsurfs[k];
          ptr[othernsurf] = facetris[icell][iface][0];
          ptr[othernsurf+1] = facetris[icell][iface][1];
          cells[othercell].nsurf += 2;
          cells[othercell].csurfs = ptr;
          tris[facetris[icell][iface][0]].id = cells[othercell].id;
          tris[facetris[icell][iface][1]].id = cells[othercell].id;
          //printf("MC add1 %d %d\n",cells[icell].id,cells[othercell].id);
          //nadd += 2;
        }

        // delete 2 tris from othercell
        // set nfacetri[othercell] = 0, so won't delete again when it is icell

        if (ntri_other == 2) {
          nfacetri[othercell][otherface] = 0;
          othernsurf = cells[othercell].nsurf;
          ptr = cells[othercell].csurfs;
          m = facetris[othercell][otherface][0];
          for (k = 0; k < othernsurf; k++)
            if (ptr[k] == m) break;
          if (k == othernsurf)
            error->one(FLERR,"Could not find surf in cleanup_MC");
          cells[othercell].csurfs[k] = cells[othercell].csurfs[othernsurf-1];
          othernsurf--;
          m = facetris[othercell][otherface][1];
          for (k = 0; k < othernsurf; k++)
            if (ptr[k] == m) break;
          if (k == othernsurf)
            error->one(FLERR,"Could not find surf in cleanup_MC");
          cells[othercell].csurfs[k] = cells[othercell].csurfs[othernsurf-1];
          othernsurf--;
          cells[othercell].nsurf -= 2;
          //printf("MC del1 %d %d\n",cells[icell].id,cells[othercell].id);
          //ndel += 2;
        }

        // delete 2 tris from icell

        ptr = cells[icell].csurfs;
        m = facetris[icell][iface][0];
        for (k = 0; k < nsurf; k++)
          if (ptr[k] == m) break;
        if (k == nsurf) error->one(FLERR,"Could not find surf in cleanup_MC");
        cells[icell].csurfs[k] = cells[icell].csurfs[nsurf-1];
        nsurf--;
        m = facetris[icell][iface][1];
        for (k = 0; k < nsurf; k++)
          if (ptr[k] == m) break;
        if (k == nsurf) error->one(FLERR,"Could not find surf in cleanup_MC");
        cells[icell].csurfs[k] = cells[icell].csurfs[nsurf-1];
        nsurf--;
        cells[icell].nsurf -= 2;
        //printf("MC dele %d %d\n",cells[icell].id,cells[othercell].id);
        //ndel += 2;

        // add 4 tris to delete list if both cells deleted them

        if (ntri_other == 2) {
          if (ndelete+4 > maxdelete) {
            maxdelete += DELTA;
            memory->grow(dellist,maxdelete,"readisurf:dellist");
          }
          dellist[ndelete++] = facetris[icell][iface][0];
          dellist[ndelete++] = facetris[icell][iface][1];
          dellist[ndelete++] = facetris[othercell][otherface][0];
          dellist[ndelete++] = facetris[othercell][otherface][1];
        }

      // cell face is shared with another proc
      // send it the cell/face indices and the 2 tris,
      //   in case they need to be assigned to the other cell based on norm

      } else {
        if (nsend == maxsend) {
          maxsend += DELTA;
          proclist = (int *)
            memory->srealloc(proclist,maxsend*sizeof(int),
                             "readisurf:proclist");
          bufsend = (SendDatum *)
            memory->srealloc(bufsend,maxsend*sizeof(SendDatum),
                             "readisurf:bufsend");
        }
        proclist[nsend] = otherproc;
        bufsend[nsend].sendcell = icell;
        bufsend[nsend].sendface = iface;
        bufsend[nsend].othercell = otherlocal;
        bufsend[nsend].otherface = otherface;
        bufsend[nsend].inwardnorm = inwardnorm;
        memcpy(&bufsend[nsend].tri1,&tris[facetris[icell][iface][0]],
               sizeof(Surf::Tri));
        memcpy(&bufsend[nsend].tri2,&tris[facetris[icell][iface][1]],
               sizeof(Surf::Tri));
        nsend++;

        // if not inwardnorm, delete 2 tris from this cell
        // also add them to delete list

        if (!inwardnorm) {
          ptr = cells[icell].csurfs;
          m = facetris[icell][iface][0];
          for (k = 0; k < nsurf; k++)
            if (ptr[k] == m) break;
          if (k == nsurf) error->one(FLERR,"Could not find surf in cleanup_MC");
          cells[icell].csurfs[k] = cells[icell].csurfs[nsurf-1];
          nsurf--;
          m = facetris[icell][iface][1];
          for (k = 0; k < nsurf; k++)
            if (ptr[k] == m) break;
          if (k == nsurf) error->one(FLERR,"Could not find surf in cleanup_MC");
          cells[icell].csurfs[k] = cells[icell].csurfs[nsurf-1];
          nsurf--;
          cells[icell].nsurf -= 2;
          //ndel += 2;

          if (ndelete+2 > maxdelete) {
            maxdelete += DELTA;
            memory->grow(dellist,maxdelete,"readisurf:dellist");
          }
          dellist[ndelete++] = facetris[icell][iface][0];
          dellist[ndelete++] = facetris[icell][iface][1];
        }
      }
    }
  }

  // perform irregular communication of list of cell faces and tri pairs

  Irregular *irregular = new Irregular(sparta);
  int nrecv = irregular->create_data_uniform(nsend,proclist,1);

  SendDatum *bufrecv = (SendDatum *)
    memory->smalloc(nrecv*sizeof(SendDatum),"readisurf:bufrecv");

  irregular->exchange_uniform((char *) bufsend,sizeof(SendDatum),
                              (char *) bufrecv);
  delete irregular;
  memory->sfree(proclist);
  memory->sfree(bufsend);

  // loop over list of received face/tri info
  // if my matching face has 2 tris, delete them
  // if my matching face has 0 tris, skip or add 2 tris depending on norm

  for (i = 0; i < nrecv; i++) {
    icell = bufrecv[i].othercell;
    iface = bufrecv[i].otherface;

    // my icell is not affected, sender cell keeps its 2 tris

    if (nfacetri[icell][iface] == 0 && bufrecv[i].inwardnorm) continue;

    // add 2 tris to icell and this processor's Surf::tris list
    // set tri IDs to new owning cell, must be done after memcpy()
    // NOTE: what about tri types?

    if (nfacetri[icell][iface] == 0) {
      int nslocal = surf->nlocal;
      surf->add_tri(cells[icell].id,1,
                    bufrecv[i].tri1.p1,bufrecv[i].tri1.p2,bufrecv[i].tri1.p3);
      memcpy(&surf->tris[nslocal],&bufrecv[i].tri1,sizeof(Surf::Tri));
      surf->tris[nslocal].id = cells[icell].id;
      surf->add_tri(cells[icell].id,1,
                    bufrecv[i].tri2.p1,bufrecv[i].tri2.p2,bufrecv[i].tri2.p3);
      memcpy(&surf->tris[nslocal+1],&bufrecv[i].tri2,sizeof(Surf::Tri));
      surf->tris[nslocal+1].id = cells[icell].id;

      nsurf = cells[icell].nsurf;
      oldcsurfs = cells[icell].csurfs;
      ptr = csurfs->get(nsurf+2);
      for (k = 0; k < nsurf; k++)
        ptr[k] = oldcsurfs[k];
      ptr[nsurf] = nslocal;
      ptr[nsurf+1] = nslocal+1;
      cells[icell].nsurf += 2;
      cells[icell].csurfs = ptr;
      //nadd += 2;
    }

    // both cells have 2 tris on common face
    // need to delete my 2 tris from icell
    // sender will get similar message from me and delete
    // inwardnorm check to see if I already deleted when sent a message,
    // else delete now and add 2 tris to delete list

    if (nfacetri[icell][iface] == 2) {
      norm = tris[facetris[icell][iface][0]].norm;
      idim = iface/2;
      if (iface % 2 && norm[idim] < 0.0) inwardnorm = 1;
      else if (iface % 2 == 0 && norm[idim] > 0.0) inwardnorm = 1;
      else inwardnorm = 0;
      if (!inwardnorm) continue;

      nsurf = cells[icell].nsurf;
      ptr = cells[icell].csurfs;
      m = facetris[icell][iface][0];
      for (k = 0; k < nsurf; k++)
        if (ptr[k] == m) break;
      if (k == nsurf) error->one(FLERR,"Could not find surf in cleanup_MC");
      cells[icell].csurfs[k] = cells[icell].csurfs[nsurf-1];
      nsurf--;
      m = facetris[icell][iface][1];
      for (k = 0; k < nsurf; k++)
        if (ptr[k] == m) break;
      if (k == nsurf) error->one(FLERR,"Could not find surf in cleanup_MC");
      cells[icell].csurfs[k] = cells[icell].csurfs[nsurf-1];
      nsurf--;
      cells[icell].nsurf -= 2;
      //ndel += 2;

      if (ndelete+2 > maxdelete) {
        maxdelete += DELTA;
        memory->grow(dellist,maxdelete,"readisurf:dellist");
      }
      dellist[ndelete++] = facetris[icell][iface][0];
      dellist[ndelete++] = facetris[icell][iface][1];
    }
  }

  memory->sfree(bufrecv);
  memory->destroy(nfacetri);
  memory->destroy(facetris);

  // compress Surf::tris list to remove deleted tris
  // must sort dellist, so as to compress tris in DESCENDING index order
  // descending, not ascending, so that a surf is not moved from end-of-list
  //   that is flagged for later deletion
  // must repoint one location in cells->csurfs to moved surf
  //   requires grid hash to find owning cell of moved surf
  // note that ghost surfs exist at this point, but caller will clear them

  if (!grid->hashfilled) grid->rehash();

  qsort(dellist,ndelete,sizeof(int),compare_indices);

  tris = surf->tris;
  int nslocal = surf->nlocal;
  for (i = 0; i < ndelete; i++) {
    m = dellist[i];
    if (m != nslocal-1) memcpy(&tris[m],&tris[nslocal-1],sizeof(Surf::Tri));
    nslocal--;

    icell = (*grid->hash)[tris[m].id];
    nsurf = cells[icell].nsurf;
    ptr = cells[icell].csurfs;
    for (k = 0; k < nsurf; k++)
      if (ptr[k] == nslocal) {
        ptr[k] = m;
        break;
      }
    if (k == nsurf) error->one(FLERR,"Did not find moved tri in cleanup_MC()");
  }

  surf->nlocal = nslocal;
  memory->destroy(dellist);

}

/* ----------------------------------------------------------------------
   adding triangles
------------------------------------------------------------------------- */

int MarchingCubes::add_triangle(int *trig, int n)
{
  for(int t = 0; t < 3*n; t++) {
    switch (trig[t]) {
    case 0:
      pt[t][0] = interpolate(v[0],v[1],lo[0],hi[0]);
      pt[t][1] = lo[1];
      pt[t][2] = lo[2];
      break;
    case 1:
      pt[t][0] = hi[0];
      pt[t][1] = interpolate(v[1],v[2],lo[1],hi[1]);
      pt[t][2] = lo[2];
      break;
    case 2:
      pt[t][0] = interpolate(v[3],v[2],lo[0],hi[0]);
      pt[t][1] = hi[1];
      pt[t][2] = lo[2];
      break;
    case 3:
      pt[t][0] = lo[0];
      pt[t][1] = interpolate(v[0],v[3],lo[1],hi[1]);
      pt[t][2] = lo[2];
      break;
    case 4:
      pt[t][0] = interpolate(v[4],v[5],lo[0],hi[0]);
      pt[t][1] = lo[1];
      pt[t][2] = hi[2];
      break;
    case 5:
      pt[t][0] = hi[0];
      pt[t][1] = interpolate(v[5],v[6],lo[1],hi[1]);
      pt[t][2] = hi[2];
      break;
    case 6:
      pt[t][0] = interpolate(v[7],v[6],lo[0],hi[0]);
      pt[t][1] = hi[1];
      pt[t][2] = hi[2];
      break;
    case 7:
      pt[t][0] = lo[0];
      pt[t][1] = interpolate(v[4],v[7],lo[1],hi[1]);
      pt[t][2] = hi[2];
      break;
    case 8:
      pt[t][0] = lo[0];
      pt[t][1] = lo[1];
      pt[t][2] = interpolate(v[0],v[4],lo[2],hi[2]);
      break;
    case 9:
      pt[t][0] = hi[0];
      pt[t][1] = lo[1];
      pt[t][2] = interpolate(v[1],v[5],lo[2],hi[2]);
      break;
    case 10:
      pt[t][0] = hi[0];
      pt[t][1] = hi[1];
      pt[t][2] = interpolate(v[2],v[6],lo[2],hi[2]);
      break;
    case 11:
      pt[t][0] = lo[0];
      pt[t][1] = hi[1];
      pt[t][2] = interpolate(v[3],v[7],lo[2],hi[2]);
      break;
    case 12: {
      int u = 0;
      pt[t][0] = pt[t][1] = pt[t][2] = 0.0;
      if (bit0 ^ bit1) {
        ++u;
        pt[t][0] += interpolate(v[0],v[1],lo[0],hi[0]);
        pt[t][1] += lo[1];
        pt[t][2] += lo[2];
      }
      if (bit1 ^ bit2) {
        ++u;
        pt[t][0] += hi[0];
        pt[t][1] += interpolate(v[1],v[2],lo[1],hi[1]);
        pt[t][2] += lo[2];
      }
      if (bit2 ^ bit3) {
        ++u;
        pt[t][0] += interpolate(v[3],v[2],lo[0],hi[0]);
        pt[t][1] += hi[1];
        pt[t][2] += lo[2];
      }
      if (bit3 ^ bit0) {
        ++u;
        pt[t][0] += lo[0];
        pt[t][1] += interpolate(v[0],v[3],lo[1],hi[1]);
        pt[t][2] += lo[2];
      }
      if (bit4 ^ bit5) {
        ++u;
        pt[t][0] += interpolate(v[4],v[5],lo[0],hi[0]);
        pt[t][1] += lo[1];
        pt[t][2] += hi[2];
      }
      if (bit5 ^ bit6) {
        ++u;
        pt[t][0] += hi[0];
        pt[t][1] += interpolate(v[5],v[6],lo[1],hi[1]);
        pt[t][2] += hi[2];
      }
      if (bit6 ^ bit7) {
        ++u;
        pt[t][0] += interpolate(v[7],v[6],lo[0],hi[0]);
        pt[t][1] += hi[1];
        pt[t][2] += hi[2];
      }
      if (bit7 ^ bit4) {
        ++u;
        pt[t][0] += lo[0];
        pt[t][1] += interpolate(v[4],v[7],lo[1],hi[1]);
        pt[t][2] += hi[2];
      }
      if (bit0 ^ bit4) {
        ++u;
        pt[t][0] += lo[0];
        pt[t][1] += lo[1];
        pt[t][2] += interpolate(v[0],v[4],lo[2],hi[2]);
      }
      if (bit1 ^ bit5) {
        ++u;
        pt[t][0] += hi[0];
        pt[t][1] += lo[1];
        pt[t][2] += interpolate(v[1],v[5],lo[2],hi[2]);
      }
      if (bit2 ^ bit6) {
        ++u;
        pt[t][0] += hi[0];
        pt[t][1] += hi[1];
        pt[t][2] += interpolate(v[2],v[6],lo[2],hi[2]);
      }
      if (bit3 ^ bit7) {
        ++u;
        pt[t][0] += lo[0];
        pt[t][1] += hi[1];
        pt[t][2] += interpolate(v[3],v[7],lo[2],hi[2]);
      }

      pt[t][0] /= static_cast<double> (u);
      pt[t][1] /= static_cast<double> (u);
      pt[t][2] /= static_cast<double> (u);
      break;
    }

    default:
      break;
    }
  }

  return n;
}

/* ----------------------------------------------------------------------
   adding triangles
------------------------------------------------------------------------- */

int MarchingCubes::add_triangle_inner(int *trig, int n)
{
  for(int t = 0; t < 3*n; t++) {
    switch (trig[t]) {
    case 0:
      pt[t][0] = i0;
      pt[t][1] = lo[1];
      pt[t][2] = lo[2];
      break;
    case 1:
      pt[t][0] = hi[0];
      pt[t][1] = i1;
      pt[t][2] = lo[2];
      break;
    case 2:
      pt[t][0] = i2;
      pt[t][1] = hi[1];
      pt[t][2] = lo[2];
      break;
    case 3:
      pt[t][0] = lo[0];
      pt[t][1] = i3;
      pt[t][2] = lo[2];
      break;
    case 4:
      pt[t][0] = i4;
      pt[t][1] = lo[1];
      pt[t][2] = hi[2];
      break;
    case 5:
      pt[t][0] = hi[0];
      pt[t][1] = i5;
      pt[t][2] = hi[2];
      break;
    case 6:
      pt[t][0] = i6;
      pt[t][1] = hi[1];
      pt[t][2] = hi[2];
      break;
    case 7:
      pt[t][0] = lo[0];
      pt[t][1] = i7;
      pt[t][2] = hi[2];
      break;
    case 8:
      pt[t][0] = lo[0];
      pt[t][1] = lo[1];
      pt[t][2] = i8;
      break;
    case 9:
      pt[t][0] = hi[0];
      pt[t][1] = lo[1];
      pt[t][2] = i9;
      break;
    case 10:
      pt[t][0] = hi[0];
      pt[t][1] = hi[1];
      pt[t][2] = i10;
      break;
    case 11:
      pt[t][0] = lo[0];
      pt[t][1] = hi[1];
      pt[t][2] = i11;
      break;
    case 12: {
      int u = 0;
      pt[t][0] = pt[t][1] = pt[t][2] = 0.0;
      if (bit0 ^ bit1) {
        ++u;
        pt[t][0] += i0;
        pt[t][1] += lo[1];
        pt[t][2] += lo[2];
      }
      if (bit1 ^ bit2) {
        ++u;
        pt[t][0] += hi[0];
        pt[t][1] += i1;
        pt[t][2] += lo[2];
      }
      if (bit2 ^ bit3) {
        ++u;
        pt[t][0] += i2;
        pt[t][1] += hi[1];
        pt[t][2] += lo[2];
      }
      if (bit3 ^ bit0) {
        ++u;
        pt[t][0] += lo[0];
        pt[t][1] += i3;
        pt[t][2] += lo[2];
      }
      if (bit4 ^ bit5) {
        ++u;
        pt[t][0] += i4;
        pt[t][1] += lo[1];
        pt[t][2] += hi[2];
      }
      if (bit5 ^ bit6) {
        ++u;
        pt[t][0] += hi[0];
        pt[t][1] += i5;
        pt[t][2] += hi[2];
      }
      if (bit6 ^ bit7) {
        ++u;
        pt[t][0] += i6;
        pt[t][1] += hi[1];
        pt[t][2] += hi[2];
      }
      if (bit7 ^ bit4) {
        ++u;
        pt[t][0] += lo[0];
        pt[t][1] += i7;
        pt[t][2] += hi[2];
      }
      if (bit0 ^ bit4) {
        ++u;
        pt[t][0] += lo[0];
        pt[t][1] += lo[1];
        pt[t][2] += i8;
      }
      if (bit1 ^ bit5) {
        ++u;
        pt[t][0] += hi[0];
        pt[t][1] += lo[1];
        pt[t][2] += i9;
      }
      if (bit2 ^ bit6) {
        ++u;
        pt[t][0] += hi[0];
        pt[t][1] += hi[1];
        pt[t][2] += i10;
      }
      if (bit3 ^ bit7) {
        ++u;
        pt[t][0] += lo[0];
        pt[t][1] += hi[1];
        pt[t][2] += i11;
      }

      pt[t][0] /= static_cast<double> (u);
      pt[t][1] /= static_cast<double> (u);
      pt[t][2] /= static_cast<double> (u);
      break;
    }

    default:
      break;
    }
  }

  return n;
}

/* ----------------------------------------------------------------------
   test a face
   if face > 0 return true if the face contains a part of the surface
------------------------------------------------------------------------- */

bool MarchingCubes::test_face(int face)
{
  double A,B,C,D;

  switch (face) {
  case -1:
  case 1:
    A = viso[0];
    B = viso[4];
    C = viso[5];
    D = viso[1];
    break;
  case -2:
  case 2:
    A = viso[1];
    B = viso[5];
    C = viso[6];
    D = viso[2];
    break;
  case -3:
  case 3:
    A = viso[2];
    B = viso[6];
    C = viso[7];
    D = viso[3];
    break;
  case -4:
  case 4:
    A = viso[3];
    B = viso[7];
    C = viso[4];
    D = viso[0];
    break;
  case -5:
  case 5:
    A = viso[0];
    B = viso[3];
    C = viso[2];
    D = viso[1];
    break;
  case -6:
  case 6:
    A = viso[4];
    B = viso[7];
    C = viso[6];
    D = viso[5];
    break;

  default:
    A = B = C = D = 0.0;
    print_cube();
    error->one(FLERR,"Invalid face code");
  };

  if (fabs(A*C - B*D) < EPSILON) return face >= 0;
  return face * A * (A*C - B*D) >= 0 ;  // face and A invert signs
}

/* ----------------------------------------------------------------------
   test a face
   if face > 0 return true if the face contains a part of the surface
------------------------------------------------------------------------- */

bool MarchingCubes::test_face_inner(int face)
{
  double A,B,C,D;
  double AB, AD, BC, DC;
  double mat[4][4], b[4], phi[4];

  // set up linear system

  switch (face) {
  case -1:
  case 1:
    A = viso[0];
    B = viso[4];
    C = viso[5];
    D = viso[1];
    AD = i0u;
    AB = i8u;
    BC = i4u;
    DC = i9u;
    break;
  case -2:
  case 2:
    A = viso[1];
    B = viso[5];
    C = viso[6];
    D = viso[2];
    AD = i1u;
    AB = i9u;
    BC = i5u;
    DC = i10u;
    break;
  case -3:
  case 3:
    A = viso[2];
    B = viso[6];
    C = viso[7];
    D = viso[3];
    AD = i2u;
    AB = i10u;
    BC = i6u;
    DC = i11u;
    break;
  case -4:
  case 4:
    A = viso[3];
    B = viso[7];
    C = viso[4];
    D = viso[0];
    AD = i3u;
    AB = i11u;
    BC = i7u;
    DC = i8u;
    break;
  case -5:
  case 5:
    A = viso[0];
    B = viso[3];
    C = viso[2];
    D = viso[1];
    AD = i0u;
    AB = i3u;
    BC = i2u;
    DC = i1u;
    break;
  case -6:
  case 6:
    A = viso[4];
    B = viso[7];
    C = viso[6];
    D = viso[5];
    AD = i4u;
    AB = i7u;
    BC = i6u;
    DC = i5u;
    break;

  default:
    A = B = C = D = 0.0;
    print_cube();
    error->one(FLERR,"Invalid face code");
  };

  // if the matrix is not singular, solve for corresponding
  // corner value points; otherwise, use avreage values

  /*if ( !(fabs(AD-BC)<0.01 || fabs(AB-DC)<0.01) &&
       !(fabs(AD-AB)<0.01 || fabs(BC-DC)<0.01) ) {

    for (int i = 0; i < 4; i++) {
      b[i] = thresh;
      for (int j = 0; j < 4; j++) mat[i][j] = 0.0;
    }

    mat[0][0] = 1.0-AD;
    mat[0][1] = AD;
    mat[1][0] = 1.0-AB;
    mat[1][2] = AB;
    mat[2][2] = 1.0-BC;
    mat[2][3] = BC;
    mat[3][1] = 1.0-DC;
    mat[3][3] = DC;

    int nosol = MathExtra::mldivide4(mat, b, phi);
    if (!nosol) {
      A = phi[0];
      B = phi[1];
      C = phi[2];
      D = phi[3];
    } else
      error->one(FLERR,"Cannot find corresponding corner values");
  }*/

  if (fabs(A*C - B*D) < EPSILON) return face >= 0;
  return face * A * (A*C - B*D) >= 0 ;  // face and A invert signs
}

/* ----------------------------------------------------------------------
   test the interior of a cube
   icase = case of the active cube in [0..15]
   if s ==  7, return true if the interior is empty
   if s == -7, return false if the interior is empty
------------------------------------------------------------------------- */

bool MarchingCubes::test_interior(int s, int icase)
{
  double t,a,b,At=0.0,Bt=0.0,Ct=0.0,Dt=0.0;
  int test = 0;
  int edge = -1;   // reference edge of the triangulation

  switch (icase) {
  case  4 :
  case 10 :
    a = ( viso[4] - viso[0] ) * ( viso[6] - viso[2] ) -
      ( viso[7] - viso[3] ) * ( viso[5] - viso[1] ) ;
    b =  viso[2] * ( viso[4] - viso[0] ) + viso[0] * ( viso[6] - viso[2] ) -
      viso[1] * ( viso[7] - viso[3] ) - viso[3] * ( viso[5] - viso[1] ) ;
    t = - b / (2*a) ;
    if (t < 0 || t > 1) return s>0 ;

    At = viso[0] + ( viso[4] - viso[0] ) * t ;
    Bt = viso[3] + ( viso[7] - viso[3] ) * t ;
    Ct = viso[2] + ( viso[6] - viso[2] ) * t ;
    Dt = viso[1] + ( viso[5] - viso[1] ) * t ;
    break ;

  case  6 :
  case  7 :
  case 12 :
  case 13 :
    switch( icase ) {
    case  6 : edge = test6 [config][2] ; break ;
    case  7 : edge = test7 [config][4] ; break ;
    case 12 : edge = test12[config][3] ; break ;
    case 13 : edge = tiling13_5_1[config][subconfig][0] ; break ;
    }
    switch( edge ) {
    case  0 :
      t  = viso[0] / ( viso[0] - viso[1] ) ;
      At = 0.0 ;
      Bt = viso[3] + ( viso[2] - viso[3] ) * t ;
      Ct = viso[7] + ( viso[6] - viso[7] ) * t ;
      Dt = viso[4] + ( viso[5] - viso[4] ) * t ;
      break ;
    case  1 :
      t  = viso[1] / ( viso[1] - viso[2] ) ;
      At = 0.0 ;
      Bt = viso[0] + ( viso[3] - viso[0] ) * t ;
      Ct = viso[4] + ( viso[7] - viso[4] ) * t ;
      Dt = viso[5] + ( viso[6] - viso[5] ) * t ;
      break ;
    case  2 :
      t  = viso[2] / ( viso[2] - viso[3] ) ;
      At = 0.0 ;
      Bt = viso[1] + ( viso[0] - viso[1] ) * t ;
      Ct = viso[5] + ( viso[4] - viso[5] ) * t ;
      Dt = viso[6] + ( viso[7] - viso[6] ) * t ;
      break ;
    case  3 :
      t  = viso[3] / ( viso[3] - viso[0] ) ;
      At = 0.0 ;
      Bt = viso[2] + ( viso[1] - viso[2] ) * t ;
      Ct = viso[6] + ( viso[5] - viso[6] ) * t ;
      Dt = viso[7] + ( viso[4] - viso[7] ) * t ;
      break ;
    case  4 :
      t  = viso[4] / ( viso[4] - viso[5] ) ;
      At = 0.0 ;
      Bt = viso[7] + ( viso[6] - viso[7] ) * t ;
      Ct = viso[3] + ( viso[2] - viso[3] ) * t ;
      Dt = viso[0] + ( viso[1] - viso[0] ) * t ;
      break ;
    case  5 :
      t  = viso[5] / ( viso[5] - viso[6] ) ;
      At = 0.0 ;
      Bt = viso[4] + ( viso[7] - viso[4] ) * t ;
      Ct = viso[0] + ( viso[3] - viso[0] ) * t ;
      Dt = viso[1] + ( viso[2] - viso[1] ) * t ;
      break ;
    case  6 :
      t  = viso[6] / ( viso[6] - viso[7] ) ;
      At = 0.0 ;
      Bt = viso[5] + ( viso[4] - viso[5] ) * t ;
      Ct = viso[1] + ( viso[0] - viso[1] ) * t ;
      Dt = viso[2] + ( viso[3] - viso[2] ) * t ;
      break ;
    case  7 :
      t  = viso[7] / ( viso[7] - viso[4] ) ;
      At = 0.0 ;
      Bt = viso[6] + ( viso[5] - viso[6] ) * t ;
      Ct = viso[2] + ( viso[1] - viso[2] ) * t ;
      Dt = viso[3] + ( viso[0] - viso[3] ) * t ;
      break ;
    case  8 :
      t  = viso[0] / ( viso[0] - viso[4] ) ;
      At = 0.0 ;
      Bt = viso[3] + ( viso[7] - viso[3] ) * t ;
      Ct = viso[2] + ( viso[6] - viso[2] ) * t ;
      Dt = viso[1] + ( viso[5] - viso[1] ) * t ;
      break ;
    case  9 :
      t  = viso[1] / ( viso[1] - viso[5] ) ;
      At = 0.0 ;
      Bt = viso[0] + ( viso[4] - viso[0] ) * t ;
      Ct = viso[3] + ( viso[7] - viso[3] ) * t ;
      Dt = viso[2] + ( viso[6] - viso[2] ) * t ;
      break ;
    case 10 :
      t  = viso[2] / ( viso[2] - viso[6] ) ;
      At = 0.0 ;
      Bt = viso[1] + ( viso[5] - viso[1] ) * t ;
      Ct = viso[0] + ( viso[4] - viso[0] ) * t ;
      Dt = viso[3] + ( viso[7] - viso[3] ) * t ;
      break ;
    case 11 :
      t  = viso[3] / ( viso[3] - viso[7] ) ;
      At = 0.0 ;
      Bt = viso[2] + ( viso[6] - viso[2] ) * t ;
      Ct = viso[1] + ( viso[5] - viso[1] ) * t ;
      Dt = viso[0] + ( viso[4] - viso[0] ) * t ;
      break ;

    default:
      print_cube();
      error->one(FLERR,"Marching cubes - invalid edge");
    }
    break;

  default:
    print_cube();
    error->one(FLERR,"Marching cubes - invalid ambiguous case");
  }

  if (At >= 0.0) test ++;
  if (Bt >= 0.0) test += 2;
  if (Ct >= 0.0) test += 4;
  if (Dt >= 0.0) test += 8;
  switch (test) {
  case  0: return s>0;
  case  1: return s>0;
  case  2: return s>0;
  case  3: return s>0;
  case  4: return s>0;
  case  5:
    if (At * Ct - Bt * Dt <  EPSILON) return s>0;
    break;
  case  6: return s>0;
  case  7: return s<0;
  case  8: return s>0;
  case  9: return s>0;
  case 10:
    if (At * Ct - Bt * Dt >= EPSILON) return s>0;
    break;
  case 11: return s<0;
  case 12: return s>0;
  case 13: return s<0;
  case 14: return s<0;
  case 15: return s<0;
  }

  return s<0;
}

/* ---------------------------------------------------------------------- */

bool MarchingCubes::modified_test_interior(int s, int icase)
{
  int edge = -1;
  int amb_face;

  int inter_amb = 0;

  switch (icase) {
  case 4:
    amb_face = 1;
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    amb_face = 2;
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    amb_face = 5;
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    if (inter_amb == 0) return false;
    else                return true;
    break;

  case 6:
    amb_face = abs(test6[config][0]);

    edge = interior_ambiguity(amb_face, s);
    inter_amb = interior_ambiguity_verification(edge);

    if (inter_amb == 0) return false;
    else                return true;

    break;

  case 7:
    s = s * -1;

    amb_face = 1;
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    amb_face = 2;
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    amb_face = 5;
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    if (inter_amb == 0) return false;
    else                return true;
    break;

  case 10:
    amb_face = abs(test10[config][0]);

    edge = interior_ambiguity(amb_face, s);
    inter_amb = interior_ambiguity_verification(edge);

    if (inter_amb == 0) return false;
    else                return true;
    break;

  case 12:
    amb_face = abs(test12[config][0]);
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);


    amb_face = abs(test12[config][1]);
    edge = interior_ambiguity(amb_face, s);
    inter_amb += interior_ambiguity_verification(edge);

    if (inter_amb == 0) return false;
    else                return true;
    break;
  }

  // should never reach here

  return true;
}

/* ---------------------------------------------------------------------- */

int MarchingCubes::interior_ambiguity(int amb_face, int s)
{
  int edge;

  switch (amb_face) {
  case 1:
  case 3:
    if (((viso[1] * s) > 0) && ((viso[7] * s) > 0)) edge = 4;
    if (((viso[0] * s) > 0) && ((viso[6] * s) > 0)) edge = 5;
    if (((viso[3] * s) > 0) && ((viso[5] * s) > 0)) edge = 6;
    if (((viso[2] * s) > 0) && ((viso[4] * s) > 0)) edge = 7;
    break;

  case 2:
  case 4:
    if (((viso[1] * s) > 0) && ((viso[7] * s) > 0)) edge = 0;
    if (((viso[2] * s) > 0) && ((viso[4] * s) > 0)) edge = 1;
    if (((viso[3] * s) > 0) && ((viso[5] * s) > 0)) edge = 2;
    if (((viso[0] * s) > 0) && ((viso[6] * s) > 0)) edge = 3;
    break;

  case 5:
  case 6:
  case 0:
    if (((viso[0] * s) > 0) && ((viso[6] * s) > 0)) edge = 8;
    if (((viso[1] * s) > 0) && ((viso[7] * s) > 0)) edge = 9;
    if (((viso[2] * s) > 0) && ((viso[4] * s) > 0)) edge = 10;
    if (((viso[3] * s) > 0) && ((viso[5] * s) > 0)) edge = 11;
    break;
  }

  return edge;
}

/* ---------------------------------------------------------------------- */

int MarchingCubes::interior_ambiguity_verification(int edge)
{
  double t, At = 0.0, Bt = 0.0, Ct = 0.0, Dt = 0.0, a = 0.0, b = 0.0;
  double verify;

  switch (edge) {

  case 0:
    a = (viso[0] - viso[1]) * (viso[7] - viso[6])
      - (viso[4] - viso[5]) * (viso[3] - viso[2]);
    b = viso[6] * (viso[0] - viso[1]) + viso[1] * (viso[7] - viso[6])
      - viso[2] * (viso[4] - viso[5])
      - viso[5] * (viso[3] - viso[2]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = viso[1] + (viso[0] - viso[1]) * t;
    Bt = viso[5] + (viso[4] - viso[5]) * t;
    Ct = viso[6] + (viso[7] - viso[6]) * t;
    Dt = viso[2] + (viso[3] - viso[2]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;

    break;

  case 1:
    a = (viso[3] - viso[2]) * (viso[4] - viso[5])
      - (viso[0] - viso[1]) * (viso[7] - viso[6]);
    b = viso[5] * (viso[3] - viso[2]) + viso[2] * (viso[4] - viso[5])
      - viso[6] * (viso[0] - viso[1])
      - viso[1] * (viso[7] - viso[6]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = viso[2] + (viso[3] - viso[2]) * t;
    Bt = viso[1] + (viso[0] - viso[1]) * t;
    Ct = viso[5] + (viso[4] - viso[5]) * t;
    Dt = viso[6] + (viso[7] - viso[6]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 2:
    a = (viso[2] - viso[3]) * (viso[5] - viso[4])
      - (viso[6] - viso[7]) * (viso[1] - viso[0]);
    b = viso[4] * (viso[2] - viso[3]) + viso[3] * (viso[5] - viso[4])
      - viso[0] * (viso[6] - viso[7])
      - viso[7] * (viso[1] - viso[0]);
    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = viso[3] + (viso[2] - viso[3]) * t;
    Bt = viso[7] + (viso[6] - viso[7]) * t;
    Ct = viso[4] + (viso[5] - viso[4]) * t;
    Dt = viso[0] + (viso[1] - viso[0]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 3:
    a = (viso[1] - viso[0]) * (viso[6] - viso[7])
      - (viso[2] - viso[3]) * (viso[5] - viso[4]);
    b = viso[7] * (viso[1] - viso[0]) + viso[0] * (viso[6] - viso[7])
      - viso[4] * (viso[2] - viso[3])
      - viso[3] * (viso[5] - viso[4]);
    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = viso[0] + (viso[1] - viso[0]) * t;
    Bt = viso[3] + (viso[2] - viso[3]) * t;
    Ct = viso[7] + (viso[6] - viso[7]) * t;
    Dt = viso[4] + (viso[5] - viso[4]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 4:

    a = (viso[2] - viso[1]) * (viso[7] - viso[4])
      - (viso[3] - viso[0]) * (viso[6] - viso[5]);
    b = viso[4] * (viso[2] - viso[1]) + viso[1] * (viso[7] - viso[4])
      - viso[5] * (viso[3] - viso[0])
      - viso[0] * (viso[6] - viso[5]);

    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = viso[1] + (viso[2] - viso[1]) * t;
    Bt = viso[0] + (viso[3] - viso[0]) * t;
    Ct = viso[4] + (viso[7] - viso[4]) * t;
    Dt = viso[5] + (viso[6] - viso[5]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 5:

    a = (viso[3] - viso[0]) * (viso[6] - viso[5])
      - (viso[2] - viso[1]) * (viso[7] - viso[4]);
    b = viso[5] * (viso[3] - viso[0]) + viso[0] * (viso[6] - viso[5])
      - viso[4] * (viso[2] - viso[1])
      - viso[1] * (viso[7] - viso[4]);
    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = viso[0] + (viso[3] - viso[0]) * t;
    Bt = viso[1] + (viso[2] - viso[1]) * t;
    Ct = viso[5] + (viso[6] - viso[5]) * t;
    Dt = viso[4] + (viso[7] - viso[4]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 6:
    a = (viso[0] - viso[3]) * (viso[5] - viso[6])
      - (viso[4] - viso[7]) * (viso[1] - viso[2]);
    b = viso[6] * (viso[0] - viso[3]) + viso[3] * (viso[5] - viso[6])
      - viso[2] * (viso[4] - viso[7])
      - viso[7] * (viso[1] - viso[2]);
    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = viso[3] + (viso[0] - viso[3]) * t;
    Bt = viso[7] + (viso[4] - viso[7]) * t;
    Ct = viso[6] + (viso[5] - viso[6]) * t;
    Dt = viso[2] + (viso[1] - viso[2]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 7:
    a = (viso[1] - viso[2]) * (viso[4] - viso[7])
      - (viso[0] - viso[3]) * (viso[5] - viso[6]);
    b = viso[7] * (viso[1] - viso[2]) + viso[2] * (viso[4] - viso[7])
      - viso[6] * (viso[0] - viso[3])
      - viso[3] * (viso[5] - viso[6]);
    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = viso[2] + (viso[1] - viso[2]) * t;
    Bt = viso[3] + (viso[0] - viso[3]) * t;
    Ct = viso[7] + (viso[4] - viso[7]) * t;
    Dt = viso[6] + (viso[5] - viso[6]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 8:
    a = (viso[4] - viso[0]) * (viso[6] - viso[2])
      - (viso[7] - viso[3]) * (viso[5] - viso[1]);
    b = viso[2] * (viso[4] - viso[0]) + viso[0] * (viso[6] - viso[2])
      - viso[1] * (viso[7] - viso[3])
      - viso[3] * (viso[5] - viso[1]);
    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = viso[0] + (viso[4] - viso[0]) * t;
    Bt = viso[3] + (viso[7] - viso[3]) * t;
    Ct = viso[2] + (viso[6] - viso[2]) * t;
    Dt = viso[1] + (viso[5] - viso[1]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 9:
    a = (viso[5] - viso[1]) * (viso[7] - viso[3])
      - (viso[4] - viso[0]) * (viso[6] - viso[2]);
    b = viso[3] * (viso[5] - viso[1]) + viso[1] * (viso[7] - viso[3])
      - viso[2] * (viso[4] - viso[0])
      - viso[0] * (viso[6] - viso[2]);
    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = viso[1] + (viso[5] - viso[1]) * t;
    Bt = viso[0] + (viso[4] - viso[0]) * t;
    Ct = viso[3] + (viso[7] - viso[3]) * t;
    Dt = viso[2] + (viso[6] - viso[2]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 10:
    a = (viso[6] - viso[2]) * (viso[4] - viso[0])
      - (viso[5] - viso[1]) * (viso[7] - viso[3]);
    b = viso[0] * (viso[6] - viso[2]) + viso[2] * (viso[4] - viso[0])
      - viso[3] * (viso[5] - viso[1])
      - viso[1] * (viso[7] - viso[3]);
    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = viso[2] + (viso[6] - viso[2]) * t;
    Bt = viso[1] + (viso[5] - viso[1]) * t;
    Ct = viso[0] + (viso[4] - viso[0]) * t;
    Dt = viso[3] + (viso[7] - viso[3]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;

  case 11:
    a = (viso[7] - viso[3]) * (viso[5] - viso[1])
      - (viso[6] - viso[2]) * (viso[4] - viso[0]);
    b = viso[1] * (viso[7] - viso[3]) + viso[3] * (viso[5] - viso[1])
      - viso[0] * (viso[6] - viso[2])
      - viso[2] * (viso[4] - viso[0]);
    if (a > 0)
      return 1;

    t = -b / (2 * a);
    if (t < 0 || t > 1)
      return 1;

    At = viso[3] + (viso[7] - viso[3]) * t;
    Bt = viso[2] + (viso[6] - viso[2]) * t;
    Ct = viso[1] + (viso[5] - viso[1]) * t;
    Dt = viso[0] + (viso[4] - viso[0]) * t;

    verify = At * Ct - Bt * Dt;

    if (verify > 0)
      return 0;
    if (verify < 0)
      return 1;
    break;
  }

  // should never reach here

  return 1;
}

/* ----------------------------------------------------------------------
   return true if the interior is empty (two faces)
------------------------------------------------------------------------- */

bool MarchingCubes::interior_test_case13()
{
  double t1, t2, At1 = 0.0, Bt1 = 0.0, Ct1 = 0.0, Dt1 = 0.0;
  double At2 = 0.0, Bt2 = 0.0, Ct2 = 0.0, Dt2 = 0.0, a = 0.0, b = 0.0, c = 0.0;

  a = (viso[0] - viso[1]) * (viso[7] - viso[6])
    - (viso[4] - viso[5]) * (viso[3] - viso[2]);
  b = viso[6] * (viso[0] - viso[1]) + viso[1] * (viso[7] - viso[6])
    - viso[2] * (viso[4] - viso[5])
    - viso[5] * (viso[3] - viso[2]);
  c = viso[1]*viso[6] - viso[5]*viso[2];

  double delta = b*b - 4*a*c;

  t1 = (-b + sqrt(delta))/(2*a);
  t2 = (-b - sqrt(delta))/(2*a);

  // DEBUG
  // printf("delta = %f, t1 = %f, t2 = %f\n", delta, t1, t2);

  if ((t1 < 1)&&(t1>0) &&(t2 < 1)&&(t2 > 0)) {
    At1 = viso[1] + (viso[0] - viso[1]) * t1;
    Bt1 = viso[5] + (viso[4] - viso[5]) * t1;
    Ct1 = viso[6] + (viso[7] - viso[6]) * t1;
    Dt1 = viso[2] + (viso[3] - viso[2]) * t1;

    double x1 = (At1 - Dt1)/(At1 + Ct1 - Bt1 - Dt1);
    double y1 = (At1 - Bt1)/(At1 + Ct1 - Bt1 - Dt1);

    At2 = viso[1] + (viso[0] - viso[1]) * t2;
    Bt2 = viso[5] + (viso[4] - viso[5]) * t2;
    Ct2 = viso[6] + (viso[7] - viso[6]) * t2;
    Dt2 = viso[2] + (viso[3] - viso[2]) * t2;

    double x2 = (At2 - Dt2)/(At2 + Ct2 - Bt2 - Dt2);
    double y2 = (At2 - Bt2)/(At2 + Ct2 - Bt2 - Dt2);

    if ((x1 < 1)&&(x1>0) &&(x2 < 1)&&(x2 > 0) &&
        (y1 < 1)&&(y1>0) &&(y2 < 1)&&(y2 > 0)) return false;
  }

  return true;
}

/* ----------------------------------------------------------------------
   comparison function invoked by qsort() called by cleanup()
   used to sort the dellist of removed tris into DESCENDING order
   this is not a class method
------------------------------------------------------------------------- */

int compare_indices(const void *iptr, const void *jptr)
{
  int i = *((int *) iptr);
  int j = *((int *) jptr);
  if (i < j) return 1;
  if (i > j) return -1;
  return 0;
}

/* ----------------------------------------------------------------------
   print cube for debugging
------------------------------------------------------------------------- */

void MarchingCubes::print_cube()
{
  fprintf(screen,"\t %d %d %d %d %d %d %d %d\n",
         v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);
}
