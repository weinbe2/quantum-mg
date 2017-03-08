// Copyright (c) 2017 Evan S Weinberg
// Header file for a multigrid object, which contains everything (?)
// needed for a multigrid preconditioner.

// NOT YET COMPLETED, WILL NOT COMPILE

// MG requires:
// * Knowledge of the number of levels.
// * Knowledge of each lattice (stored as a std::vector of Lattice2D*)
// * Knowledge of each transfer object (stored as a std::vector of TransferMG*)
// * Knowledge of the fine level stencil (stored as the first element of
//    a std::vector of Stencil2D*)
// ** Optionally knows lower levels, if they've been constructed. (If there,
//     it's stored in the above std::vector of Stencil2D*. If not there, stored
//     as a zero pointer in the vector.)
// * Is written such that one level is pushed at a time.
// ** This allows the user to start storing levels in the MG object when
//     recursively generating coarser levels.
// * A private function to (optionaly) explicitly build the coarse stencil.
//     NOT YET IMPLEMENTED YET.
// ** Implemented as a flag on pushing a new level.
// ** Should have a flag to specify if it should be built from the
//     original operator or right block Jacobi operator.
//     NOT YET MEANINGFULLY IMPLEMENTED YET.
// * A function to apply the stencil at a specified level.
// * A function to prepare, solve, reconstruct right block Jacobi and, where
//    possible, Schur preconditioned systems. 
//    NOT YET IMPLEMENTED YET.
// * Optional, but for convenience, two pre-allocated temporary vectors
//    at each level.
// ** This avoids allocating and deallocating temporary vectors, such as for
//     the preconditioned functions.
// * Optional, but for convenience, the ability to store non-block-orthogonalized
//    null vectors (perhaps for updating or projecting null vectors)

#ifndef QMG_MULTIGRID_OBJECT
#define QMG_MULTIGRID_OBJECT

#include <complex>
#include <cmath>
#include <iostream>
#include <vector>

using std::complex;
using std::vector;

// QLINALG
#include "blas/generic_vector.h"

// QMG
#include "lattice/lattice.h"
#include "stencil/stencil_2d.h"
#include "transfer/transfer.h"

class MultigridMG
{
private:
  // Get rid of copy, assignment operator.
  MultigridMG(MultigridMG const &);
  MultigridMG& operator=(MultigridMG const &);

  // Current number of levels. Gets updated when user
  // adds another layer. There's a public function that
  // exposes this. 
  int num_levels;

  // Knowledge of each lattice. Should have length = num_levels.
  vector<Lattice2D*> lattice_list;

  // Knowledge of each transfer operator. Should have length = num_levels - 1.
  vector<TransferMG*> transfer_list;

  // Knowledge of each stencil. Required to contain fine level stencil,
  // optionally contains coarser stencils (otherwise the element will be
  // set as a zero pointer). Should have length = num_levels.
  vector<Stencil2D*> stencil_list;

  // Knowledge of if the MG class has created the stencil itself
  // or not. This informs what needs to be deleted in the destructor.
  // Should have length = num_levels, and is_stencil_managed[0] = false always.
  vector<bool> is_stencil_managed; 

  // Optional, but for convenience, two pre-allocated temporary vectors
  // at each level. Should have length = num_levels.
  vector<complex<double>*> temp_vec_1;
  vector<complex<double>*> temp_vec_2;

  // Optionally, but for convenience, the ability to store non-block-
  // orthogonalized null vectors. Should have length = num_levels - 1.
  vector<complex<double>**> global_null_vectors; 

public:

  // Enum for the stencil preconditioned application type.
  enum QMGMultigridPrecondStencil
  {
    QMG_MULTIGRID_PRECOND_ORIGINAL = 0, // Original stencil
    QMG_MULTIGRID_PRECOND_RIGHT_BLOCK_JACOBI = 1, // Right block jacobi stencil.
    QMG_MULTIGRID_PRECOND_SCHUR = 2, // Schur decomposed right block jacobi stencil.
  };
  
public:
  
  // Constructor. Takes in fine lattice and stencil.
  // At some point, I should add a constructor which can take multiple levels
  // at once. 
  MultigridMG(Lattice2D* in_lat, Stencil2D* in_stencil)
  {
    // Initialize number of levels to 1.
    num_levels = 1;

    // Push the fine lattice onto lattice_list.
    lattice_list.push_back(in_lat);

    // Prepare the first two temporary vectors. 
    temp_vec_1.push_back(allocate_vector<complex<double>>(lattice_list[0]->get_size_cv()));
    temp_vec_2.push_back(allocate_vector<complex<double>>(lattice_list[0]->get_size_cv()));

    // Push the fine stencil.
    stencil_list.push_back(in_stencil);

    // Push that we are NOT responsible for the first stencil.
    is_stencil_managed.push_back(false);
  }

  // Destructor. Clean up!
  ~MultigridMG()
  {
    int i;

    // Clean up temporary vectors.
    for (i = 0; i < num_levels; i++)
    {

      // Deallocate temporary vectors.
      if (temp_vec_1[i] != 0) { deallocate_vector(&temp_vec_1[i]); }
      if (temp_vec_2[i] != 0) { deallocate_vector(&temp_vec_2[i]); }
      
      // Clean up stencils that this class created.
      if (is_stencil_managed[i] && stencil_list[i] != 0)
      {
        delete stencil_list[i];
        stencil_list[i] = 0;
      }
    }

    // Safely clean up null vectors.
    for (i = 0; i < num_levels-1; i++)
    {
      int num_null = lattice_list[i+1]->get_nc();
      if (global_null_vectors[i] != 0)
      {
        for (int j = 0; j < num_null; j++)
        {
          if (global_null_vectors[i][j] != 0)
          {
            deallocate_vector(&global_null_vectors[i][j]);
            global_null_vectors[i][j] = 0;
          }
        }
        delete[] global_null_vectors[i];
        global_null_vectors[i] = 0;
      }
    }
  }

  // Public function exposing number of levels.
  inline int get_num_levels()
  {
    return num_levels;
  }

  // Public function exposing a given lattice.
  inline Lattice2D* get_lattice(int i)
  {
    if (i >= 0 && i < num_levels)
    {
      return lattice_list[i];
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Lattice level " << i << " does not exist in MultigridMG object.\n";
      return 0;
    }
  }

  // Public function exposing a transfer between level i and i+1.
  inline TransferMG* get_transfer(int i)
  {
    if (i >= 0 && i < num_levels-1)
    {
      return transfer_list[i];
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Transfer object level " << i << " does not exist in MultigridMG object.\n";
      return 0;
    }
  }

  // Public function exposing a stencil between level i and i+1.
  inline Stencil2D* get_stencil(int i)
  {
    if (i >= 0 && i < num_levels)
    {
      return stencil_list[i];
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Stencil object level " << i << " does not exist in MultigridMG object.\n";
      return 0;
    }
  }

  // Public function copying saved null vectors into a given array of vectors.
  void get_global_null_vectors(int i, complex<double>** out_vectors)
  {
    if (i >= 0 && i < num_levels-1)
    {
      if (out_vectors != 0 && global_null_vectors[i] != 0)
      {
        for (int j = 0; j < lattice_list[i+1]->get_nc(); j++)
        {
          if (out_vectors[j] != 0 && global_null_vectors[i][j] != 0)
          {
            copy_vector(out_vectors[j], global_null_vectors[i][j], lattice_list[i]->get_size_cv());
          }
        }
      }
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Null vectors level " << i << " does not exist in MultigridMG object.\n";
      return;
    }
  }

  // Public function to push a new level!
  // Arg 1: New lattice object.
  // Arg 2: New transfer object.
  // Arg 3: (Not yet implemented) If we should explicitly build the new coarse stencil or not.
  // Arg 4: (Not yet implemented) If we should build the new coarse stencil from a preconditioned
  //          version of the stencil a level up.
  // Arg 5: Copy in the global null vectors. These are null vectors
  //          that are not yet block orthonormalized.
  void push_level(Lattice2D* new_lat, TransferMG* new_transfer, bool build_stencil = false, QMGMultigridPrecondStencil build_stencil_from = QMG_MULTIGRID_PRECOND_ORIGINAL, complex<double>** nvecs = 0)
  {
    // Update number of levels.
    num_levels++;

    // Push new lattice.
    lattice_list.push_back(new_lat);

    // Push new transfer object.
    transfer_list.push_back(new_transfer);

    // Allocate new temporary vectors.
    temp_vec_1.push_back(allocate_vector<complex<double>>(new_lat->get_size_cv()));
    temp_vec_2.push_back(allocate_vector<complex<double>>(new_lat->get_size_cv()));

    // Deal with stencil.
    if (build_stencil)
    {
      cout << "[QMG-ERROR]: MultigridMG does not support building a stencil yet. Default to null.\n";
      stencil_list.push_back(0);
      is_stencil_managed.push_back(false); // will become true
    }
    else
    {
      stencil_list.push_back(0);
      is_stencil_managed.push_back(false);
    }

    // What operator are we building the stencil from?
    // At some point we'll at least support building it from
    // the right block jacobi stencil...
    if (build_stencil_from != QMG_MULTIGRID_PRECOND_ORIGINAL)
    {
      cout << "[QMG-ERROR]: MultigridMG does not support preconditioned stencils yet. Default to original stencil.\n";
    }

    // Copy global null vectors, if they're non-zero.
    if (nvecs != 0)
    {
      global_null_vectors.push_back(new complex<double>*[new_lat->get_nc()]);
      for (int j = 0; j < new_lat->get_nc(); j++)
      {
        if (nvecs[j] != 0)
        {
          complex<double>* tmp = allocate_vector<complex<double>>(lattice_list[num_levels-2]->get_size_cv());
          copy_vector(tmp, nvecs[j], lattice_list[num_levels-2]->get_size_cv());
          global_null_vectors[num_levels-2][j] = tmp;
        }
      }
    }
    else
    {
      global_null_vectors.push_back(0);
    }
  }

  // A flavor of the function to push a new level
  // that assumes we're not building the coarse stencil,
  // but still wants to save global null vectors.
  // Arg 1: New lattice object.
  // Arg 2: New transfer object.
  // Arg 3: Copy in the global null vectors. These are null vectors
  //          that are not yet block orthonormalized.
  void push_level(Lattice2D* new_lat, TransferMG* new_transfer,complex<double>** nvecs)
  {
    push_level(new_lat, new_transfer, false, QMG_MULTIGRID_PRECOND_ORIGINAL, nvecs);
  }



  // A function that applies the stencil at a given level. Will apply
  // the stencil if it exists, otherwise it'll "emulate" it via
  // prolong, apply stencil, restrict, recursively if needed.
  // That form is more for experimenting than for performance.
  // At some point it'll take preconditioner flags...
  // In math form, applies lhs = M rhs.
  void apply_stencil(complex<double>* lhs, complex<double>* rhs, int i)
  {
    if (i >= 0 && i < num_levels)
    {
      if (stencil_list[i] != 0)
      {
        stencil_list[i]->apply_M(lhs, rhs);
      }
      else
      {
        // Recurse!
        // This is why we carry around temporary vectors.

        // Zero out a vector to prolong into, and a vector we apply into.
        zero_vector(temp_vec_1[i-1], lattice_list[i-1]->get_size_cv());
        zero_vector(temp_vec_2[i-1], lattice_list[i-1]->get_size_cv());
        

        // Prolong rhs into the temporary vector.
        transfer_list[i-1]->prolong_c2f(rhs, temp_vec_1[i-1]);

        // Call the stencil one level up.
        apply_stencil(temp_vec_2[i-1], temp_vec_1[i-1], i-1);

        // Restrict temporary vector into lhs.
        transfer_list[i-1]->restrict_f2c(temp_vec_2[i-1], lhs);
      }
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot apply stencil at level " << i << "\n";
      return;
    }
  }

  // A function that prolongs from a coarse to a fine level at a given fine level.
  void prolong_c2f(complex<double>* coarse_cv, complex<double>* fine_cv, int i)
  {
    if (i >= 0 && i < num_levels-1)
    {
      transfer_list[i]->prolong_c2f(coarse_cv, fine_cv);
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot apply prolong at level " << i << "\n";
      return;
    }
  }

  // A function that restricts from a fine to a coarse level at a given fine level.
  void restrict_f2c(complex<double>* fine_cv, complex<double>* coarse_cv, int i)
  {
    if (i >= 0 && i < num_levels-1)
    {
      transfer_list[i]->restrict_f2c(fine_cv, coarse_cv);
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot apply prolong at level " << i << "\n";
      return;
    }
  }
};

#endif // QMG_MULTIGRID_OBJECT
