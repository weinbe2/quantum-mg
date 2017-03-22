// Copyright (c) 2017 Evan S Weinberg
// A test of the transfer class. Goes between a fine and coarse 2d lattice.
// Intentionally made to follow the structure of
// prolong_restrict_test.

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

int main(int argc, char** argv)
{
  // Set output precision to be long.
  cout << setprecision(20);

  // Random number generator.
  std::mt19937 generator (1337u);

  // Basic information for fine level.
  const int x_len = 64;
  const int y_len = 64;
  const int dof = 1; // can set to whatever.

  // Blocking size.
  const int x_block = 4;
  const int y_block = 4;

  // Number of null vectors (equiv. degrees of freedom on coarse level)
  const int coarse_dof = 2;

  // Create a lattice object for the fine lattice.
  Lattice2D* lat = new Lattice2D(x_len, y_len, dof);

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
