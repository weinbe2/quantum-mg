// Copyright (c) 2017 Evan S Weinberg
// Create a coarse operator from another stencil and a transfer object. 

#ifndef QMG_COARSE
#define QMG_COARSE

#include <iostream>
#include <complex>
using std::complex;

#include "blas/generic_vector.h"
#include "lattice/lattice.h"
#include "stencil/stencil_2d.h"
#include "transfer/transfer.h"

// Build a coarse operator from another stencil and a
// a transfer object. 
struct CoarseOperator2D : public Stencil2D
{
protected:
  // Get rid of copy, assignment.
  CoarseOperator2D(CoarseOperator2D const &);
  CoarseOperator2D& operator=(CoarseOperator2D const &);

  // Temporary storage.
  complex<double>* tmp_coarse;
  complex<double>* tmp_fine;
  complex<double>* tmp_Afine;

  // Save the fine lattice.
  Lattice2D* fine_lat; 

public:

  // Base constructor.
  // NOTE! Need a way to determine QMG_PIECE_... based on the
  // input stencil. Maybe there needs to be a function in each stencil
  // to determine the size, then some smart function that figures
  // out what the stencil will look like after coarsening?
  // Also need some smart way to deal with the mass (for \gamma_5 ops)
  // Currently this function only transfers identity shifts.
  CoarseOperator2D(Lattice2D* in_lat, Stencil2D* fine_stencil, Lattice2D* fine_lattice, TransferMG* transfer)
    : Stencil2D(in_lat, QMG_PIECE_CLOVER_HOPPING, fine_stencil->get_shift(), 0.0, 0.0), fine_lat(fine_lattice)
  {
    const int coarse_vol = lat->get_volume();
    const int coarse_size = lat->get_size_cv();
    const int coarse_nc = lat->get_nc();
    const int fine_size = fine_lat->get_size_cv();

    // Allocate temporary space.
    tmp_coarse = allocate_vector<complex<double>>(coarse_size);
    tmp_fine = allocate_vector<complex<double>>(fine_size);
    tmp_Afine = allocate_vector<complex<double>>(fine_size);

    ///////////////////////////////////
    // Step 0: Transfer shifts over. //
    ///////////////////////////////////

    // We need some set of flags concerning transfering
    // eo and dof shifts over...

    ///////////////////////////////////////////////////////////////////////////
    // Step 1: learn about (some of) the coarse clover from the fine clover. //
    ///////////////////////////////////////////////////////////////////////////

    zero_vector(clover, lat->get_size_cm());
    
    // Take matrix elements of the fine clover term.
    for (int color = 0; color < coarse_nc; color++)
    {
      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      if (coarse_vol == 1)
      {
        tmp_coarse[lat->cv_coord_to_index(0, color)] = 1.0;
        transfer->prolong_c2f(tmp_coarse, tmp_fine);
        fine_stencil->apply_M_clover(tmp_Afine, tmp_fine);
        zero_vector(tmp_coarse, coarse_size);
        transfer->restrict_f2c(tmp_Afine, tmp_coarse);
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(0, c, color)] += tmp_coarse[lat->cv_coord_to_index(0,c)];
        continue; 
      }

      // Set a 1 at each coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol; i++)
        tmp_coarse[lat->cv_coord_to_index(i, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_clover(tmp_Afine, tmp_fine);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // Fill matrix elements into clover.
      for (int i = 0; i < coarse_vol; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];
    }

    //////////////////////////////////////////////////////////////////
    // Step 2: learn about some of the coarse clover (all if it's a //
    //         distance one fine stencil), and some of the hopping  //
    //         term (all if it's a distance one fine stencil)       //
    //////////////////////////////////////////////////////////////////

    // Take matrix elements of the fine hopping term. This gets tweaked
    // if the coarse lattice has dimension 1 or 2 in some direction.

    zero_vector(hopping, lat->get_size_hopping());

    for (int color = 0; color < coarse_nc; color++)
    {
      

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // If the coarse volume is 1, there's no meaningful
      // hopping term.
      if (coarse_vol == 1)
      {
        tmp_coarse[lat->cv_coord_to_index(0, color)] = 1.0;
        transfer->prolong_c2f(tmp_coarse, tmp_fine);
        fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine);
        zero_vector(tmp_coarse, coarse_size);
        transfer->restrict_f2c(tmp_Afine, tmp_coarse);
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(0, c, color)] += tmp_coarse[lat->cv_coord_to_index(0,c)];
        continue; 
      }

      /////////////////////////////////////////////////
      // Learn from the +x part of the fine stencil. //
      /////////////////////////////////////////////////

      // Set a 1 at each even coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_XP1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // even coarse sites update coarse clover, odd coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

      if (lat->get_dim_mu(0) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i+coarse_vol/2, c, color, QMG_DIR_INDEX_XP1)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each odd coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_XP1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // odd coarse sites update coarse clover, even coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      if (lat->get_dim_mu(0) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i, c, color, QMG_DIR_INDEX_XP1)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

      /////////////////////////////////////////////////
      // Learn from the +y part of the fine stencil. //
      /////////////////////////////////////////////////

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each even coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_YP1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // even coarse sites update coarse clover, odd coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

      if (lat->get_dim_mu(1) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i+coarse_vol/2, c, color, QMG_DIR_INDEX_YP1)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each odd coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_YP1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // odd coarse sites update coarse clover, even coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      if (lat->get_dim_mu(1) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i, c, color, QMG_DIR_INDEX_YP1)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

      /////////////////////////////////////////////////
      // Learn from the -x part of the fine stencil. //
      /////////////////////////////////////////////////

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each even coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_XM1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // even coarse sites update coarse clover, odd coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

      if (lat->get_dim_mu(0) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i+coarse_vol/2, c, color, QMG_DIR_INDEX_XM1)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each odd coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_XM1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // odd coarse sites update coarse clover, even coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      if (lat->get_dim_mu(0) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i, c, color, QMG_DIR_INDEX_XM1)] += tmp_coarse[lat->cv_coord_to_index(i, c)];


      /////////////////////////////////////////////////
      // Learn from the -y part of the fine stencil. //
      /////////////////////////////////////////////////

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each even coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_YM1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // even coarse sites update coarse clover, odd coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

      if (lat->get_dim_mu(1) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i+coarse_vol/2, c, color, QMG_DIR_INDEX_YM1)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      zero_vector(tmp_coarse, coarse_size);
      zero_vector(tmp_fine, fine_size);
      zero_vector(tmp_Afine, fine_size);

      // Set a 1 at each odd coarse site, at dof 'color'
      for (int i = 0; i < coarse_vol/2; i++)
        tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, color)] = 1.0;

      // Prolong, apply clover, restrict. 
      transfer->prolong_c2f(tmp_coarse, tmp_fine);
      fine_stencil->apply_M_hopping(tmp_Afine, tmp_fine, QMG_DIR_INDEX_YM1);
      zero_vector(tmp_coarse, coarse_size);
      transfer->restrict_f2c(tmp_Afine, tmp_coarse);

      // odd coarse sites update coarse clover, even coarse sites update hopping.
      for (int i = 0; i < coarse_vol/2; i++)
        for (int c = 0; c < coarse_nc; c++)
          clover[lat->cm_coord_to_index(i+coarse_vol/2, c, color)] += tmp_coarse[lat->cv_coord_to_index(i+coarse_vol/2, c)];

      if (lat->get_dim_mu(1) == 1)
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            clover[lat->cm_coord_to_index(i, c, color)] += tmp_coarse[lat->cv_coord_to_index(i, c)];
      else
        for (int i = 0; i < coarse_vol/2; i++)
          for (int c = 0; c < coarse_nc; c++)
            hopping[lat->hopping_coord_to_index(i, c, color, QMG_DIR_INDEX_YM1)] += tmp_coarse[lat->cv_coord_to_index(i, c)];

    }

    // Still need to coarsen in 2-link, corner terms...
    // you know, when I ever actually implement them.
  }

  ~CoarseOperator2D()
  {
    deallocate_vector(&tmp_coarse);
    deallocate_vector(&tmp_fine);
    deallocate_vector(&tmp_Afine);
  }

};

#endif