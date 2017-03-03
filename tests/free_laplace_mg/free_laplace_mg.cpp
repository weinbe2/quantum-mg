// Copyright (c) 2017 Evan S Weinberg
// A test of an algebraic V-cycle for the free laplace equation.
// Just uses relaxation, so not a K-cycle!

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <random>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"

// QMG
#include "lattice/lattice.h"
#include "transfer/transfer.h"
#include "stencil/stencil_2d.h"
#include "multigrid/multigrid.h"

// Grab laplace operator (we just use unit gauge fields -> free laplace)
#include "operators/gaugedlaplace.h"
#include "u1/u1_utils.h"


int main(int argc, char** argv)
{
  // Iterators.
  int i, j; 

  // Set output precision to be long.
  cout << setprecision(20);

  // Random number generator.
  std::mt19937 generator (1337u);

  // Basic information for fine level.
  const int x_len = 64;
  const int y_len = 64;
  const int dof = 1; 

  // Information on the Laplace operator.
  const double mass = 0.1;

  // Blocking size.
  const int x_block = 2;
  const int y_block = 2;

  // Number of null vectors.
  // We're just doing the free laplace, so we only need one.
  const int coarse_dof = 1;

  // How many times to refine. 
  const int n_refine = 2; // (64 -> 32)

  // Create a lattice object for the fine lattice.
  Lattice2D** lats = new Lattice2D*[n_refine+1]
  lats[0] = new Lattice2D(x_len, y_len, dof);

  // Create a unit gauged laplace stencil.
  complex<double>* unit_gauge = allocate_vector<complex<double>>(lats[0]->get_size_gauge());
  unit_gauge_u1(unit_gauge, lats[0]);
  GaugedLaplace2D* laplace_op = new GaugedLaplace2D(lats[0], mass*mass, unit_gauge);

  // Create a MultigridMG object, push top level onto it!
  MultigridMG* mg_object = new MultigridMG(lats[0], laplace_op);

  // Create coarse lattices, unit null vectors, transfer objects.
  // Push into MultigridMG object. 
  int curr_x_len = x_len;
  int curr_y_len = y_len;
  Transfer2D** transfer_objs = new Transfer2D*[n_refine];
  for (i = 1; i <= n_refine; i++)
  {
    // Update to the new lattice size.
    curr_x_len /= x_block;
    curr_y_len /= y_block;

    // Create a new lattice object.
    lats[i] = new Lattice2D(curr_x_len, curr_y_len, coarse_dof);

    // Create a new null vector. These are copied into local memory in the
    // transfer object, so we can create and destroy these in this loop.
    complex<double>* null_vector = allocate_vector<complex<double>*>(lats[i-1]->get_size_cv());
    constant_vector(null_vector, 1.0, lats[i-1]->get_size_cv());

    // Create and populate a transfer object.
    // Fine lattice, coarse lattice, null vector(s), perform the block ortho.
    transfer_objs[i-1] = new TransferMG(lats[i-1], lats[i], &null_vector, true);

    // Push a new level on the multigrid object! Also, save the global null vector.
    // Arg 1: New lattice
    // Arg 2: New transfer object (between new and prev lattice)
    // Arg 3: Should we construct the coarse stencil? (Not supported yet.)
    // Arg 4: What should we construct the coarse stencil from? (Not relevant yet.)
    // Arg 5: Non-block-orthogonalized null vector.
    mg_object->push_level(lats[i], transfer_objs[i-1], false, QMG_MULTIGRID_PRECOND_ORIGINAL, &null_vector);

  }

  // WRITTEN UP TO HERE

  // Create a lattice object for the coarse lattice.
  Lattice2D* coarse_lat = new Lattice2D(x_len/x_block, y_len/y_block, coarse_dof);

  // For convenience.
  const int fine_size_cv = lat->get_size_cv();
  const int coarse_size_cv = coarse_lat->get_size_cv();

  // Allocate 2 null vectors.
  const int num_nvec = coarse_dof; // different names make sense in different contexts.
  complex<double>** null_vectors = new complex<double>*[num_nvec];
  null_vectors[0] = allocate_vector<complex<double>>(fine_size_cv);
  null_vectors[1] = allocate_vector<complex<double>>(fine_size_cv);

  // Create two random vectors, the first one even only, second whole lattice.
  gaussian(null_vectors[0], fine_size_cv/2, generator);
  zero_vector(null_vectors[0] + fine_size_cv/2, fine_size_cv/2);
  gaussian(null_vectors[1], fine_size_cv, generator);


  // Create Transfer class.
  // Arg 1: Fine lattice.
  // Arg 2: Coarse lattice.
  // Arg 3: Null vectors. Can deduce number of them from coarse lattice.
  //           Copies the null vectors into internal memory.
  // Arg 4: Optional, if we should block orthonormalize. Default true.
  TransferMG transferer(lat, coarse_lat, null_vectors, true);

  // Test prolong and restrict!

  // We'll create a coarse vector and a fine vector.
  complex<double>* coarse_cv_1 = allocate_vector<complex<double>>(coarse_size_cv);
  zero_vector(coarse_cv_1, coarse_size_cv);

  complex<double>* coarse_cv_2 = allocate_vector<complex<double>>(coarse_size_cv);
  zero_vector(coarse_cv_2, coarse_size_cv);

  complex<double>* fine_cv_1 = allocate_vector<complex<double>>(fine_size_cv);
  zero_vector(fine_cv_1, fine_size_cv);

  // Set coarse vector as a random vector. 
  gaussian(coarse_cv_1, coarse_size_cv, generator);

  // Check norm of original vector.
  cout << "Norm of original coarse vector = " << sqrt(norm2sq(coarse_cv_1, coarse_size_cv)) << "\n";

  // In the Galerkin case, prolong then restrict should preserve the coarse
  // vector. Let's verify that.
  transferer.prolong_c2f(coarse_cv_1, fine_cv_1);
  transferer.restrict_f2c(fine_cv_1, coarse_cv_2);

  // Check norm, difference for prolong-restrict vector.
  cout << "Norm of vector after prolong-restrict = " << sqrt(norm2sq(coarse_cv_2, coarse_size_cv)) << "\n";
  cout << "Error between vectors is = " << sqrt(diffnorm2sq(coarse_cv_1, coarse_cv_2, coarse_size_cv)) << "\n";

  // Last step: Clean up. 
  deallocate_vector(&coarse_cv_1);
  deallocate_vector(&coarse_cv_2);
  deallocate_vector(&fine_cv_1);
  deallocate_vector(&null_vectors[0]);
  deallocate_vector(&null_vectors[1]);
  delete[] null_vectors;
  delete lat;
  delete coarse_lat;

  return 0;
}
