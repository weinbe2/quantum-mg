// Copyright (c) 2017 Evan S Weinberg
// Test of a Wilson implementation.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"
#include "verbosity/verbosity.h"
#include "inverters/inverter_struct.h"
#include "inverters/generic_bicgstab_l.h"
//#include "inverters/generic_tfqmr.h"

// QMG
#include "lattice/lattice.h"
#include "cshift/cshift_2d.h"
#include "u1/u1_utils.h"

#include "operators/wilson.h"

int main(int argc, char** argv)
{
  // Set output to not be that long.
  cout << setiosflags(ios::scientific) << setprecision(15);

  // Random number generator.
  std::mt19937 generator (1337u);

  // Basic information.
  const int x_len = 32;
  const int y_len = 32;
  const int dof = 2;

  // Staggered specific information.
  const double mass = 0.1;

  // Create a lattice object.
  Lattice2D* lat = new Lattice2D(x_len, y_len, dof);

  // Get a vector size.
  const int cv_size = lat->get_size_cv();

  // Prepare some storage.
  complex<double>* rhs = allocate_vector<complex<double>>(cv_size);
  complex<double>* lhs = allocate_vector<complex<double>>(cv_size);
  complex<double>* check = allocate_vector<complex<double>>(cv_size);

  // Zero the vectors.
  zero_vector(rhs, cv_size);
  zero_vector(lhs, cv_size);
  zero_vector(check, cv_size);

  // Prepare the gauge field.
  Lattice2D* lat_gauge = new Lattice2D(x_len, y_len, 1); // hack...
  complex<double>* gauge_field = allocate_vector<complex<double>>(lat->get_size_gauge());
  read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l32t32b100_heatbath.dat");
  //unit_gauge_u1(gauge_field, lat_gauge);
  delete lat_gauge;

  // Create a gauged laplace stencil.
  Wilson2D* wilson_stencil = new Wilson2D(lat, mass, gauge_field);

  // Try an inversion.
  cout << "[QMG-TEST]: Test a matrix inversion on an even point.\n";

  // Drop a point on the rhs on an even site, get norm.
  rhs[lat->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  double rhs_norm = sqrt(norm2sq(rhs, cv_size));

  // Set an initial guess of random noise
  // (for whatever, I nan out when I don't do this...)
  gaussian(lhs, cv_size, generator);
  /*wilson_stencil->apply_M(lhs, rhs);
  cout << lhs[lat->cv_coord_to_index(x_len/2+1, y_len/2, 0)] << "\n";
  cout << lhs[lat->cv_coord_to_index(x_len/2+1, y_len/2, 1)] << "\n";
  cout << lhs[lat->cv_coord_to_index(x_len/2, y_len/2+1, 0)] << "\n";
  cout << lhs[lat->cv_coord_to_index(x_len/2, y_len/2+1, 1)] << "\n";
  cout << lhs[lat->cv_coord_to_index(x_len/2-1, y_len/2, 0)] << "\n";
  cout << lhs[lat->cv_coord_to_index(x_len/2-1, y_len/2, 1)] << "\n";
  cout << lhs[lat->cv_coord_to_index(x_len/2, y_len/2-1, 0)] << "\n";
  cout << lhs[lat->cv_coord_to_index(x_len/2, y_len/2-1, 1)] << "\n";
  return 0;*/

  // Define inverter parameters.
  int max_iter = 4000;
  double tol = 1e-7;
  int bicgstab_l = 6;

  // Prepare verbosity struct.
  // By default prints with high verbosity, switch to VERB_SUMMARY or VERB_NONE
  // if you want to see less.
  inversion_verbose_struct* verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-BICGSTAB-6-INFO]: "));

  // Reset double output format.
  cout << setiosflags(ios::scientific) << setprecision(6);

  // Perform a BiCGstab-L inversion
  inversion_info invif = minv_vector_bicgstab_l(lhs, rhs, cv_size, max_iter, tol, bicgstab_l, apply_stencil_2D_M, (void*)wilson_stencil, verb);
  //inversion_info invif = minv_vector_tfqmr(lhs, rhs, cv_size, max_iter, tol, apply_stencil_2D_M, (void*)wilson_stencil, verb);

  // Check results.
  if (invif.success == true)
  {
    cout << "[QMG-TEST]: Algorithm " << invif.name << " took " << invif.iter
      << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  else // failed, maybe.
  {
    cout << "[QMG-TEST]: Potential Error! Algorithm " << invif.name
      << " took " << invif.iter << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  cout << "[QMG-TEST]: Computing [check] = A [lhs] as a confirmation.\n";
  // Check and make sure we get the right answer.
  wilson_stencil->apply_M(check, lhs);
  double explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative error is " << explicit_resid << "\n";
  
  // Clean up.
  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&check);
  deallocate_vector(&gauge_field);

  delete verb; 
  delete wilson_stencil;
  delete lat;

  return 0;
}
