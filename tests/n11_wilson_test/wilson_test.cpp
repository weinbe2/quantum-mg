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
#include "inverters/generic_bicgstab.h"
#include "inverters/generic_bicgstab_l.h"
#include "inverters/generic_tfqmr.h"
#include "inverters/generic_cg.h"
#include "inverters/generic_gcr.h"

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
  const double beta = 6.0;
  const int dof = 2;

  // Staggered specific information.
  const double mass = -0.055;

  bool do_free = false;

  // Create a lattice object.
  Lattice2D* lat = new Lattice2D(x_len, y_len, dof);

  // Get a vector size.
  const int cv_size = lat->get_size_cv();

  // Prepare some storage.
  complex<double>* rhs = allocate_vector<complex<double>>(cv_size);
  complex<double>* lhs = allocate_vector<complex<double>>(cv_size);
  complex<double>* init_rhs = allocate_vector<complex<double>>(cv_size);
  complex<double>* init_guess = allocate_vector<complex<double>>(cv_size);
  complex<double>* check = allocate_vector<complex<double>>(cv_size);

  // Zero the vectors.
  zero_vector(rhs, cv_size);
  zero_vector(lhs, cv_size);
  zero_vector(init_rhs, cv_size);
  gaussian(init_guess, cv_size, generator);
  zero_vector(check, cv_size);

  // Prepare the gauge field.
  Lattice2D* lat_gauge = new Lattice2D(x_len, y_len, 1); // hack...
  complex<double>* gauge_field = allocate_vector<complex<double>>(lat_gauge->get_size_gauge());
  if (do_free)
  {
    unit_gauge_u1(gauge_field, lat_gauge);
  }
  else
  {
    bool need_heatbath = false;
    if (beta == 6.0)
    {
      switch (x_len)
      {
        case 32:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l32t32b60_heatbath.dat");
          break;
        case 64:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l64t64b60_heatbath.dat");
          break;
        case 128:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l128t128b60_heatbath.dat");
          break;
        case 192:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l192t192b60_heatbath.dat");
          break;
        case 256:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l256t256b60_heatbath.dat");
          break;
        default:
          need_heatbath = true;
          break;
      }
    }
    else if (beta == 10.0)
    {
      switch (x_len)
      {
        case 32:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l32t32b100_heatbath.dat");
          break;
        case 64:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l64t64b100_heatbath.dat");
          break;
        case 128:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l128t128b100_heatbath.dat");
          break;
        case 192:
          read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l192t192b100_heatbath.dat");
          break;
        default:
          need_heatbath = true;
          break;
      }
    }
    else
      need_heatbath = true;

    if (need_heatbath)
    {
      std::cout << "[QMG-NOTE]: L = " << x_len << " beta = " << beta << " requires heatbath generation.\n";

      int n_therm = 4000; // how many heatbath steps to perform.
      int n_meas = 100; // how often to measure the plaquette, topo.
      double* phases = allocate_vector<double>(lat_gauge->get_size_gauge());
      zero_vector(phases, lat_gauge->get_size_gauge());
      double plaq = 0.0; // track along the way
      double topo = 0.0;
      for (int i = 0; i < n_therm; i += n_meas)
      {
        // Perform non-compact update.
        heatbath_noncompact_update(phases, lat_gauge, beta, n_therm/n_meas, generator);

        // Get compact links.
        polar_vector(phases, gauge_field, lat_gauge->get_size_gauge());

        plaq = std::real(get_plaquette_u1(gauge_field, lat_gauge));
        topo = std::real(get_topo_u1(gauge_field, lat_gauge));
        std::cout << "[QMG-HEATBATH]: Update " << i << " Plaq " << plaq << " Topo " << topo << "\n";
      }

      // Acquire final gauge field.
      polar_vector(phases, gauge_field, lat_gauge->get_size_gauge());

      // Clean up.
      deallocate_vector(&phases);
    }
  }
  delete lat_gauge;

  // Create a gauged laplace stencil.
  Stencil2D* stencil = new Wilson2D(lat, mass, gauge_field);
  stencil->build_dagger_stencil();

  // Define some default params.
  int max_iter = 4000;
  double tol = 1e-10;
  int bicgstab_l = 4;
  double explicit_resid = 0.0; // to be filled later.
  int restart_freq = 4;
  int restart_freq_2 = 16;

 // Drop a point on the rhs on an even site, get norm.
  init_rhs[lat->cv_coord_to_index(x_len/2, y_len/2, 0)] = 1.0;
  double rhs_norm = sqrt(norm2sq(init_rhs, cv_size));

  // Prepare verbosity struct.
  // By default prints with high verbosity, switch to VERB_SUMMARY or VERB_NONE
  // if you want to see less.
  inversion_verbose_struct* verb;
  inversion_info invif;

  //////////////
  // BiCGstab //
  //////////////
  copy_vector(lhs, init_guess, cv_size);
  copy_vector(rhs, init_rhs, cv_size);
  verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-BICGSTAB-INFO]: "));
  invif = minv_vector_bicgstab(lhs, rhs, cv_size, max_iter, tol, apply_stencil_2D_M, (void*)stencil, verb);
  // Check results.
  if (invif.success == true)
  {
    cout << "[QMG-TEST-BICGSTAB]: Algorithm " << invif.name << " took " << invif.iter
      << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  else // failed, maybe.
  {
    cout << "[QMG-TEST-BICGSTAB]: Potential Error! Algorithm " << invif.name
      << " took " << invif.iter << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  cout << "[QMG-TEST-BICGSTAB]: Computing [check] = A [lhs] as a confirmation.\n";
  // Check and make sure we get the right answer.
  zero_vector(check, cv_size);
  stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(init_rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST-BICGSTAB]: The relative error is " << explicit_resid << "\n";
  delete verb;

  /////////////////
  // BiCGstab-Ls //
  /////////////////
  copy_vector(lhs, init_guess, cv_size);
  copy_vector(rhs, init_rhs, cv_size);
  verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-BICGSTAB-4-INFO]: "));
  invif = minv_vector_bicgstab_l(lhs, rhs, cv_size, max_iter, tol, bicgstab_l, apply_stencil_2D_M, (void*)stencil, verb);
  // Check results.
  if (invif.success == true)
  {
    cout << "[QMG-TEST-BICGSTAB-4]: Algorithm " << invif.name << " took " << invif.iter
      << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  else // failed, maybe.
  {
    cout << "[QMG-TEST-BICGSTAB-4]: Potential Error! Algorithm " << invif.name
      << " took " << invif.iter << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  cout << "[QMG-TEST-BICGSTAB-4: Computing [check] = A [lhs] as a confirmation.\n";
  // Check and make sure we get the right answer.
  zero_vector(check, cv_size);
  stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(init_rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST-BICGSTAB-4]: The relative error is " << explicit_resid << "\n";
  delete verb;

  ///////////
  // TFQMR //
  ///////////
  copy_vector(lhs, init_guess, cv_size);
  copy_vector(rhs, init_rhs, cv_size);
  verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-TFQMR-INFO]: "));
  invif = minv_vector_tfqmr(lhs, rhs, cv_size, max_iter, tol, apply_stencil_2D_M, (void*)stencil, verb);
  // Check results.
  if (invif.success == true)
  {
    cout << "[QMG-TEST-TFQMR]: Algorithm " << invif.name << " took " << invif.iter
      << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  else // failed, maybe.
  {
    cout << "[QMG-TEST-TFQMR]: Potential Error! Algorithm " << invif.name
      << " took " << invif.iter << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  cout << "[QMG-TEST-TFQMR]: Computing [check] = A [lhs] as a confirmation.\n";
  // Check and make sure we get the right answer.
  zero_vector(check, cv_size);
  stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(init_rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST-TFQMR]: The relative error is " << explicit_resid << "\n";
  delete verb;

  ////////
  // CG //
  ////////
  copy_vector(lhs, init_guess, cv_size);
  copy_vector(rhs, init_rhs, cv_size);
  verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-CG-INFO]: "));
  // Prepare
  zero_vector(check, cv_size);
  stencil->prepare_M_dagger_M(check, rhs);
  copy_vector(rhs, check, cv_size);
  // Done prepare
  invif = minv_vector_cg(lhs, rhs, cv_size, max_iter, tol, apply_stencil_2D_M_dagger_M, (void*)stencil, verb);
  // Check results.
  if (invif.success == true)
  {
    cout << "[QMG-TEST-CG]: Algorithm " << invif.name << " took " << invif.iter
      << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  else // failed, maybe.
  {
    cout << "[QMG-TEST-CG]: Potential Error! Algorithm " << invif.name
      << " took " << invif.iter << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  cout << "[QMG-TEST-CG]: Computing [check] = A [lhs] as a confirmation.\n";
  // Check and make sure we get the right answer.
  zero_vector(check, cv_size);
  stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(init_rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST-CG]: The relative error is " << explicit_resid << "\n";
  delete verb;

  ////////////
  // GCR(4) //
  ////////////
  copy_vector(lhs, init_guess, cv_size);
  copy_vector(rhs, init_rhs, cv_size);
  verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-GCR(4)-INFO]: "));
  invif = minv_vector_gcr_restart(lhs, rhs, cv_size, 2*max_iter, tol, restart_freq, apply_stencil_2D_M, (void*)stencil, verb);
  // Check results.
  if (invif.success == true)
  {
    cout << "[QMG-TEST-GCR(4)]: Algorithm " << invif.name << " took " << invif.iter
      << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  else // failed, maybe.
  {
    cout << "[QMG-TEST-GCR(4)]: Potential Error! Algorithm " << invif.name
      << " took " << invif.iter << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  cout << "[QMG-TEST-GCR(4)]: Computing [check] = A [lhs] as a confirmation.\n";
  // Check and make sure we get the right answer.
  zero_vector(check, cv_size);
  stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(init_rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST-GCR(4)]: The relative error is " << explicit_resid << "\n";
  delete verb;

  ////////////
  // GCR(16) //
  ////////////
  copy_vector(lhs, init_guess, cv_size);
  copy_vector(rhs, init_rhs, cv_size);
  verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-GCR(16)-INFO]: "));
  invif = minv_vector_gcr_restart(lhs, rhs, cv_size, 2*max_iter, tol, restart_freq_2, apply_stencil_2D_M, (void*)stencil, verb);
  // Check results.
  if (invif.success == true)
  {
    cout << "[QMG-TEST-GCR(16)]: Algorithm " << invif.name << " took " << invif.iter
      << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  else // failed, maybe.
  {
    cout << "[QMG-TEST-GCR(16)]: Potential Error! Algorithm " << invif.name
      << " took " << invif.iter << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  cout << "[QMG-TEST-GCR(16)]: Computing [check] = A [lhs] as a confirmation.\n";
  // Check and make sure we get the right answer.
  zero_vector(check, cv_size);
  stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(init_rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST-GCR(16)]: The relative error is " << explicit_resid << "\n";
  delete verb;

  //////////////
  // GCR(128) //
  //////////////
  copy_vector(lhs, init_guess, cv_size);
  copy_vector(rhs, init_rhs, cv_size);
  verb = new inversion_verbose_struct(VERB_DETAIL, std::string("[QMG-TEST-GCR(128)-INFO]: "));
  invif = minv_vector_gcr_restart(lhs, rhs, cv_size, 2*max_iter, tol, 128, apply_stencil_2D_M, (void*)stencil, verb);
  // Check results.
  if (invif.success == true)
  {
    cout << "[QMG-TEST-GCR(128)]: Algorithm " << invif.name << " took " << invif.iter
      << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  else // failed, maybe.
  {
    cout << "[QMG-TEST-GCR(128)]: Potential Error! Algorithm " << invif.name
      << " took " << invif.iter << " iterations to reach a tolerance of "
      << sqrt(invif.resSq)/rhs_norm << "\n";
  }
  cout << "[QMG-TEST-GCR(128)]: Computing [check] = A [lhs] as a confirmation.\n";
  // Check and make sure we get the right answer.
  zero_vector(check, cv_size);
  stencil->apply_M(check, lhs);
  explicit_resid = sqrt(diffnorm2sq(init_rhs, check, cv_size))/rhs_norm;
  cout << "[QMG-TEST-GCR(128)]: The relative error is " << explicit_resid << "\n";
  delete verb;
  
  // Clean up.
  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&init_guess);
  deallocate_vector(&check);
  deallocate_vector(&gauge_field);

  delete stencil;
  delete lat;

  return 0;
}
