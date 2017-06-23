// Copyright (c) 2017 Evan S Weinberg
// Create a coarse operator from another stencil and a transfer object. 
// If chiral, top half of dof should be top chirality,
// bottom half should be bottom chirality.

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

  // Is it a chiral stencil?
  bool is_chiral; 

  // Is this the coarse version of a right block jacobi stencil?
  bool use_rbjacobi;

public:
  // Enum for if we should build the dagger and/or rbjacobi stencil.
  enum QMGCoarseBuildStencil
  {
    QMG_COARSE_BUILD_ORIGINAL = 0, // build coarse stencil only.
    QMG_COARSE_BUILD_DAGGER = 1, // also build dagger stencil
    QMG_COARSE_BUILD_RBJACOBI = 2, // also build rbjacobi stencil
    QMG_COARSE_BUILD_DAGGER_RBJACOBI = 3, // build both dagger, rbjacobi stencil.
  };

public:

  // Base constructor to set up a bare stencil.
  CoarseOperator2D(Lattice2D* in_lat, int pieces, bool is_chiral, complex<double> in_shift = 0.0, complex<double> in_eo_shift = 0.0, complex<double> in_dof_shift = 0.0)
    : Stencil2D(in_lat, pieces, in_shift, in_eo_shift, in_dof_shift), is_chiral(is_chiral), use_rbjacobi(false)
  { ; }

  // Base constructor to build a coarse stencil from a fine stencil.
  // NOTE! Need a way to determine QMG_PIECE_... based on the
  // input stencil. Maybe there needs to be a function in each stencil
  // to determine the size, then some smart function that figures
  // out what the stencil will look like after coarsening?
  // Also need some smart way to deal with the mass (for \gamma_5 ops)
  // Currently this function only transfers identity shifts.
  CoarseOperator2D(Lattice2D* in_lat, Stencil2D* fine_stencil, Lattice2D* fine_lattice, TransferMG* transfer, bool is_chiral = false, bool use_rbjacobi = false, QMGCoarseBuildStencil build_extra = QMG_COARSE_BUILD_ORIGINAL)
    : Stencil2D(in_lat, QMG_PIECE_CLOVER_HOPPING, 0.0, 0.0, 0.0), fine_lat(fine_lattice), is_chiral(is_chiral), use_rbjacobi(use_rbjacobi)
  {
    const int coarse_vol = lat->get_volume();
    const int coarse_size = lat->get_size_cv();
    const int coarse_nc = lat->get_nc();
    const int fine_size = fine_lat->get_size_cv();

    // Allocate temporary space.
    tmp_coarse = allocate_vector<complex<double>>(coarse_size);
    tmp_fine = allocate_vector<complex<double>>(fine_size);
    tmp_Afine = allocate_vector<complex<double>>(fine_size);

    // Prepare for rbjacobi build. 
    if (use_rbjacobi)
    {
      fine_stencil->perform_swap_rbjacobi();
    }

    ///////////////////////////////////
    // Step 0: Transfer shifts over. //
    ///////////////////////////////////

    // We need some set of flags concerning transfering
    // eo and dof shifts over...
    shift = fine_stencil->get_shift();

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

    // Undo rbjacobi build. 
    if (use_rbjacobi)
    {
      fine_stencil->perform_swap_rbjacobi();
    }

    // Build dagger, rbjacobi stencils

    if (build_extra == QMG_COARSE_BUILD_DAGGER || build_extra == QMG_COARSE_BUILD_DAGGER_RBJACOBI)
    {
      build_dagger_stencil();
    }

    if (build_extra == QMG_COARSE_BUILD_RBJACOBI || build_extra == QMG_COARSE_BUILD_DAGGER_RBJACOBI)
    {
      build_rbjacobi_stencil();
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

  // The coarse operator could have any number of dof per sites.
  static int get_dof(int i = 0)
  {
    return -1;
  }

  // The coarse operator might have a sense of chirality.
  static chirality_state has_chirality()
  {
    return QMG_CHIRAL_UNKNOWN; 
  }

  virtual void gamma5(complex<double>* vec)
  {
    if (is_chiral)
    {
      const int nc = lat->get_nc();
      {
        for (int c = nc/2; c < nc; c++)
          cax_blas(-1.0, vec+c, nc, lat->get_size_cv()/nc);
      }
    }
  }

  virtual void gamma5(complex<double>* g5_vec, complex<double>* vec)
  {
    if (is_chiral)
    {
      const int nc = lat->get_nc();
      {
        for (int c = 0; c < nc/2; c++)
          caxy_blas(1.0, vec+c, nc, g5_vec+c, nc, lat->get_size_cv()/nc);

        for (int c = nc/2; c < nc; c++)
          caxy_blas(-1.0, vec+c, nc, g5_vec+c, nc, lat->get_size_cv()/nc);
      }
    }
  }

  // Chirality either does not exist or is internal dof.
  virtual void chiral_projection(complex<double>* vector, bool is_up)
  {
    if (is_chiral)
    {
      const int nc = lat->get_nc();
      if (is_up)
      {
        for (int c = 0; c < nc/2; c++)
          zero_vector_blas(vector+nc/2+c, nc, lat->get_size_cv()/nc);
      }
      else
      {
        for (int c = 0; c < nc/2; c++)
          zero_vector_blas(vector+c, nc, lat->get_size_cv()/nc);
      }
    }
  }

  // Copy projection onto up, down.
  virtual void chiral_projection_copy(complex<double>* orig, complex<double>* dest, bool is_up)
  {
    if (is_chiral)
    {
      const int nc = lat->get_nc();
      if (is_up)
      {
        for (int c = 0; c < nc/2; c++)
        {
          copy_vector_blas(dest+c, orig+c, nc, lat->get_size_cv()/nc);
          zero_vector_blas(dest+nc/2+c, nc, lat->get_size_cv()/nc);
        }
      }
      else
      {
        for (int c = 0; c < nc/2; c++)
        {
          copy_vector_blas(dest+nc/2+c, orig+nc/2+c, nc, lat->get_size_cv()/nc);
          zero_vector_blas(dest+c, nc, lat->get_size_cv()/nc);
        }
      }
    }
  }

  // Copy the down projection into a new vector, perform the up in place.
  virtual void chiral_projection_both(complex<double>* orig_to_up, complex<double>* down)
  {
    if (is_chiral)
    {
      const int nc = lat->get_nc();
      for (int c = 0; c < nc/2; c++)
      {
        copy_vector_blas(down+nc/2+c, orig_to_up+nc/2+c, nc, lat->get_size_cv()/nc);
        zero_vector_blas(down+c, nc, lat->get_size_cv()/nc);
        zero_vector_blas(orig_to_up+nc/2+c, nc, lat->get_size_cv()/nc);
      }
    }
  }

};

#endif
