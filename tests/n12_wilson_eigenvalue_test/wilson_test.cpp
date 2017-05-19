// Copyright (c) 2017 Evan S Weinberg
// Test of a square laplace implementation.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"
#include "interfaces/arpack/generic_arpack.h"

// QMG
#include "lattice/lattice.h"
#include "cshift/cshift_2d.h"
#include "u1/u1_utils.h"

#include "operators/wilson.h"

int main(int argc, char** argv)
{
  // Set output to not be that long.
  cout << setiosflags(ios::fixed) << setprecision(6);

  // Iterators and such.

  // Basic information.
  const int x_len = 32;
  const int y_len = 32;
  const int dof = 2;

  // Staggered specific information.
  // For 64^2, beta = 6.0, eigenvalues go negative around -0.075.
  const double mass = -0.07;

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
  read_gauge_u1(gauge_field, lat_gauge, "../common_cfgs_u1/l64t64b60_heatbath.dat");
  //unit_gauge_u1(gauge_field, lat_gauge);
  delete lat_gauge; 

  // Create a gauged laplace stencil.
  Wilson2D* wilson_stencil = new Wilson2D(lat, mass, gauge_field);

  // Define some default params.
  int max_iter = 4000;
  double tol = 1e-15;

  //////////////////////////////
  // Get the entire spectrum. //
  //////////////////////////////

  arpack_dcn* arpack;

  complex<double>* eigs;
  complex<double>** evecs;

  if (cv_size <= 2048)
  {
    std::cout << "Computing entire spectrum.\n";
    arpack = new arpack_dcn(cv_size, max_iter, tol, apply_stencil_2D_M, (void*)wilson_stencil);

    eigs = new complex<double>[cv_size];
    evecs = new complex<double>*[cv_size];
    for (int i = 0; i < cv_size; i++)
    {
      evecs[i] = allocate_vector<complex<double> >(cv_size);
    }

    arpack->get_entire_eigensystem(eigs, arpack_dcn::ARPACK_SMALLEST_REAL);

    for (int i = 0; i < cv_size; i++)
      std::cout << "Eigenvalue " << i << " has value " << eigs[i] << "\n";

    for (int i = 0; i < cv_size; i++)
    {
      deallocate_vector(&evecs[i]);
    }
    delete[] evecs;
    delete[] eigs;
    delete arpack; 
  }

  ///////////////////////////////////
  // Get a subset of the spectrum. //
  ///////////////////////////////////

  // Prepare an arpack structure for grabbing some small
  // amount of eigenvalues.
  int n_eigens = 20;
  arpack = new arpack_dcn(cv_size, max_iter, tol,
                apply_stencil_2D_M, (void*)wilson_stencil,
                n_eigens, 3*n_eigens);

  std::cout << "Computing " << n_eigens << " lowest eigenvalues.\n";

  eigs = new complex<double>[n_eigens];

  if(!arpack->prepare_eigensystem(arpack_dcn::ARPACK_SMALLEST_MAGNITUDE, 20))
  {
    cout << "[ERROR]: Znaupd code: " << arpack->get_solve_info().znaupd_code << "\n";
  }
  arpack->get_eigensystem(eigs, arpack_dcn::ARPACK_SMALLEST_MAGNITUDE);

  for (int j = 0; j < n_eigens; j++)
    std::cout << eigs[j] << "\n";
  std::cout << "\n";

  delete eigs;
  delete arpack;

  // Clean up.
  deallocate_vector(&lhs);
  deallocate_vector(&rhs);
  deallocate_vector(&check);
  deallocate_vector(&gauge_field);

  delete wilson_stencil;
  delete lat;

  return 0;
}
