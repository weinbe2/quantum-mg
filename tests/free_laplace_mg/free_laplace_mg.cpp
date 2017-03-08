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
  const int n_refine = 2; // (64 -> 32 -> 16)

  // Information about the solve.

  // Solver tolerance.
  const double tol = 1e-2; 

  // Maximum iterations.
  const int max_iter = 10000;

  // Create a lattice object for the fine lattice.
  Lattice2D** lats = new Lattice2D*[n_refine+1];
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
  TransferMG** transfer_objs = new TransferMG*[n_refine];
  for (i = 1; i <= n_refine; i++)
  {
    // Update to the new lattice size.
    curr_x_len /= x_block;
    curr_y_len /= y_block;

    // Create a new lattice object.
    lats[i] = new Lattice2D(curr_x_len, curr_y_len, coarse_dof);

    // Create a new null vector. These are copied into local memory in the
    // transfer object, so we can create and destroy these in this loop.
    complex<double>* null_vector = allocate_vector<complex<double>>(lats[i-1]->get_size_cv());
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
    mg_object->push_level(lats[i], transfer_objs[i-1], false, MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL, &null_vector);

    // Clean up local vector, since they get copied in.
    deallocate_vector(&null_vector);
  }

  /////////////////////////////////////////////////////////////
  // Alright! Let's do a solve that isn't MG preconditioned. //
  /////////////////////////////////////////////////////////////

  // We're going to solve Ax = b with Richardson iterations.

  // Relaxation parameter---standard jacobi preconditioner
  // for nd-dimensional Laplace.
  const double relax_omega = 1.0/(2.0*lats[0]->get_nd()+mass*mass);

  // How many iters of relax to apply per outer loop.
  const int n_relax = 4; 

  // Create a right hand side, fill with gaussian random numbers.
  complex<double>* b = allocate_vector<complex<double>>(lats[0]->get_size_cv());
  gaussian(b, lats[0]->get_size_cv(), generator);
  double bnorm = sqrt(norm2sq(b, lats[0]->get_size_cv()));

  // Create a place to accumulate a solution. Assume a zero initial guess.
  complex<double>* x = allocate_vector<complex<double>>(lats[0]->get_size_cv());
  zero_vector(x, lats[0]->get_size_cv());

  // Create a place to compute Ax. Since we have a zero initial guess, this
  // starts as zero.
  complex<double>* Ax = allocate_vector<complex<double>>(lats[0]->get_size_cv());
  zero_vector(Ax, lats[0]->get_size_cv());

  // Create a place to store the residual. Since we have zero initial guess,
  // the initial residual is b - Ax = b.
  complex<double>* r = allocate_vector<complex<double>>(lats[0]->get_size_cv());
  copy_vector(r, b, lats[0]->get_size_cv());

  // Create a place to store the current residual norm.
  double rnorm; 

  // Create a place to store the error.
  complex<double>* e = allocate_vector<complex<double>>(lats[0]->get_size_cv());
  zero_vector(e, lats[0]->get_size_cv());

  // Relax until we get sick of it.
  for (i = 0; i < max_iter; i++)
  {
    // Zero the error.
    zero_vector(e, lats[0]->get_size_cv());

    // Relax on the residual via Richardson. (Looks like pre-smoothing.)
    // e = A^{-1} r, via n_relax hits of richardson. 
    for (j = 0; j < n_relax; j++)
    {
      zero_vector(Ax, lats[0]->get_size_cv());
      mg_object->apply_stencil(Ax, e, 0); // top level stencil.

      // e += omega(r - Ax)
      caxpbypz(relax_omega, r, -relax_omega, Ax, e, lats[0]->get_size_cv());
    }

    // Update the solution.
    cxpy(e, x, lats[0]->get_size_cv());

    // Update the residual. 
    zero_vector(Ax, lats[0]->get_size_cv());
    mg_object->apply_stencil(Ax, e, 0); // top level stencil.
    caxpy(-1.0, Ax, r, lats[0]->get_size_cv());

    // Check norm.
    rnorm = sqrt(norm2sq(r, lats[0]->get_size_cv()));
    cout << "Outer step " << i << ": tolerance " << rnorm/bnorm << "\n";
    if (rnorm/bnorm < tol)
      break; 
  }

  // Check solution.
  zero_vector(Ax, lats[0]->get_size_cv());
  mg_object->apply_stencil(Ax, x, 0);
  cout << "Check: tolerance " << sqrt(diffnorm2sq(b, Ax, lats[0]->get_size_cv()))/bnorm << "\n";

  /////////////////////////////////
  // Alright! Two level V cycle. //
  /////////////////////////////////

  // Clean up a bit.
  zero_vector(x, lats[0]->get_size_cv());
  zero_vector(Ax, lats[0]->get_size_cv());
  copy_vector(r, b, lats[0]->get_size_cv());

  // We need a second level version of the above vectors. 

  // Create a place to compute the coarse Ae.
  // Since we have a zero initial guess, this starts as zero.
  complex<double>* Ae_coarse = allocate_vector<complex<double>>(lats[1]->get_size_cv());
  zero_vector(Ae_coarse, lats[1]->get_size_cv());

  // Create a place to store the coarse residual. Since we have zero initial guess,
  // the initial residual is b - Ax = b.
  complex<double>* r_coarse = allocate_vector<complex<double>>(lats[1]->get_size_cv());
  zero_vector(r_coarse, lats[1]->get_size_cv());

  // Create a place to store the coarse error.
  complex<double>* e_coarse = allocate_vector<complex<double>>(lats[1]->get_size_cv());
  zero_vector(e_coarse, lats[1]->get_size_cv());

  // Create a place to prolong the coarse error into.
  complex<double>* e_coarse_pro = allocate_vector<complex<double>>(lats[0]->get_size_cv());
  zero_vector(e_coarse_pro, lats[0]->get_size_cv());

  // Relax with a simple 2-level V cycle until we get sick of it.
  for (i = 0; i < max_iter; i++)
  {
    // Zero the error.
    zero_vector(e, lats[0]->get_size_cv());

    // Relax on the residual via Richardson. (Looks like pre-smoothing.)
    // e = A^{-1} r, via n_relax hits of richardson. 
    for (j = 0; j < n_relax; j++)
    {
      zero_vector(Ax, lats[0]->get_size_cv());
      mg_object->apply_stencil(Ax, e, 0); // top level stencil.

      // e += omega(r - Ax)
      caxpbypz(relax_omega, r, -relax_omega, Ax, e, lats[0]->get_size_cv());
    }

    // Update the solution.
    cxpy(e, x, lats[0]->get_size_cv());

    // Update the residual. 
    zero_vector(Ax, lats[0]->get_size_cv());
    mg_object->apply_stencil(Ax, e, 0); // top level stencil.
    caxpy(-1.0, Ax, r, lats[0]->get_size_cv());


    // Go to coarse level. 

    // Restrict the residual. 
    zero_vector(r_coarse, lats[1]->get_size_cv());
    mg_object->restrict_f2c(r, r_coarse, 0);

    // Zero the coarse error.
    zero_vector(e_coarse, lats[1]->get_size_cv());

    // Relax on the coarse residual via Richardson.
    for (j = 0; j < n_relax; j++)
    {
      zero_vector(Ae_coarse, lats[1]->get_size_cv());
      mg_object->apply_stencil(Ae_coarse, e_coarse, 1); // first level down stencil.

      // e += omega(r - Ae)
      caxpbypz(relax_omega, r_coarse, -relax_omega, Ae_coarse, e_coarse, lats[1]->get_size_cv());
    }

    // Prolong the error.
    zero_vector(e_coarse_pro, lats[0]->get_size_cv());
    mg_object->prolong_c2f(e_coarse, e_coarse_pro, 0);

    // Update the solution (normally we'd post-smooth, but this is just a V-cycle.)
    cxpy(e_coarse_pro, x, lats[0]->get_size_cv());

    // Update the residual.
    zero_vector(Ax, lats[0]->get_size_cv());
    mg_object->apply_stencil(Ax, e_coarse_pro, 0); // top level stencil.
    caxpy(-1.0, Ax, r, lats[0]->get_size_cv());

    // Check norm.
    rnorm = sqrt(norm2sq(r, lats[0]->get_size_cv()));
    cout << "V-cycle Outer step " << i << ": tolerance " << rnorm/bnorm << "\n";
    if (rnorm/bnorm < tol)
      break; 
  }

  // Check solution.
  zero_vector(Ax, lats[0]->get_size_cv());
  mg_object->apply_stencil(Ax, x, 0);
  cout << "V-cycle Check: tolerance " << sqrt(diffnorm2sq(b, Ax, lats[0]->get_size_cv()))/bnorm << "\n";

  ///////////////
  // Clean up. //
  ///////////////

  deallocate_vector(&Ae_coarse);
  deallocate_vector(&e_coarse);
  deallocate_vector(&r_coarse);
  deallocate_vector(&e_coarse_pro);

  deallocate_vector(&e);
  deallocate_vector(&r);
  deallocate_vector(&Ax);
  deallocate_vector(&x);
  deallocate_vector(&b);

  // Delete MultigridMG.
  delete mg_object;

  // Delete transfer objects.
  for (i = 0; i < n_refine; i++)
  {
    delete transfer_objs[i];
  }
  delete[] transfer_objs;

  // Delete stencil.
  delete laplace_op;

  // Delete lattices.
  for (i = 0; i <= n_refine; i++)
  {
    delete lats[i];
  }
  delete[] lats; 

  return 0;
}
