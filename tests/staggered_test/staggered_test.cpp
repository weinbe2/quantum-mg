// Copyright (c) 2017 Evan S Weinberg
// Test of a square laplace implementation.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"
#include "verbosity/verbosity.h"
#include "inverters/inverter_struct.h"
#include "inverters/generic_cg.h"
#include "inverters/generic_gcr.h"

// QMG
#include "lattice/lattice.h"
#include "cshift/cshift_2d.h"
#include "u1/u1_utils.h"

#include "operators/staggered.h"

int main(int argc, char** argv)
{
  // Set output to not be that long.
  cout << setiosflags(ios::fixed) << setprecision(6);

  // Iterators and such.

  // Basic information.
  const int x_len = 32;
  const int y_len = 32;
  const int dof = 1;

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
  complex<double>* gauge_field = allocate_vector<complex<double>>(lat->get_size_gauge());
  read_gauge_u1(gauge_field, lat, "../common_cfgs_u1/l32t32b60_heatbath.dat");
  //unit_gauge_u1(gauge_field, lat);

  // Create a gauged laplace stencil.
  Staggered2D* stag_stencil = new Staggered2D(lat, mass, gauge_field);

  // Try an inversion.
  cout << "[QMG-TEST]: Test a matrix inversion on an even point.\n";

  // Drop a point on the rhs on an even site, get norm.
  rhs[lat->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  double rhs_norm = sqrt(norm2sq(rhs, cv_size));

  // Define inverter parameters.
  int max_iter = 4000;
  double tol = 1e-7;

  // Prepare verbosity struct.
  // By default prints with high verbosity, switch to VERB_SUMMARY or VERB_NONE
  // if you want to see less.
  inversion_verbose_struct* verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-GCR-INFO]: "));

  // Reset double output format.
  cout << setiosflags(ios::scientific) << setprecision(6);

  // Perform a GCR inversion.
  inversion_info invif = minv_vector_gcr(lhs, rhs, cv_size, max_iter, tol, apply_stencil_2D_M, (void*)stag_stencil, verb);

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
  stag_stencil->apply_M(check, lhs);
  double explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative error is " << explicit_resid << "\n";
  
  // Try an e-o preconditioned solve.
  cout << "[QMG-TEST]: Test an even-odd preconditioned solve.\n";

  // Zero the vectors.
  zero_vector(rhs, cv_size);
  zero_vector(lhs, cv_size);
  zero_vector(check, cv_size);

  // Drop a few points on the rhs.
  rhs[lat->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  rhs[lat->cv_coord_to_index(x_len/2+1, y_len/2, 0)] = 1.0;
  rhs_norm = sqrt(norm2sq(rhs, cv_size));

  // Prepare. Use 'check' has a temporary b_new.
  stag_stencil->prepare_b(check, rhs);

  // Perform a CG inversion.
  verb->verb_prefix = std::string("[QMG-TEST-CG-INFO]: ");
  invif = minv_vector_cg(lhs, check, cv_size/2, max_iter, tol, apply_eo_staggered_2D_M, (void*)stag_stencil, verb);

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

  // Check and make sure we get the right answer. Reconstruct first.
  stag_stencil->reconstruct_x(lhs, rhs);
  zero_vector(check, cv_size);
  stag_stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST]: The relative error is " << explicit_resid << "\n";

  // Clean up.
  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&check);
  deallocate_vector(&gauge_field);

  delete verb; 
  delete stag_stencil;
  delete lat;

  return 0;
}