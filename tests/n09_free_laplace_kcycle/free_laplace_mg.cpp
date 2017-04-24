// Copyright (c) 2017 Evan S Weinberg
// A test of a geometric K-cycle for the free laplace equation.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"
#include "inverters/generic_gcr.h"
#include "inverters/generic_gcr_var_precond.h"

// QMG
#include "lattice/lattice.h"
#include "transfer/transfer.h"
#include "stencil/stencil_2d.h"
#include "multigrid/stateful_multigrid.h"

// Grab laplace operator (we just use unit gauge fields -> free laplace)
#include "operators/gaugedlaplace.h"
#include "u1/u1_utils.h"




int main(int argc, char** argv)
{
  // Iterators.
  int i;

  // Set output precision to be long.
  cout << setprecision(20);

  // Random number generator.
  std::mt19937 generator (1337u);

  // Basic information for fine level.
  const int x_len = 128;
  const int y_len = 128;
  const int dof = GaugedLaplace2D::get_dof(); 

  // Information on the Laplace operator.
  const double mass = 0.001;

  // Blocking size.
  const int x_block = 2;
  const int y_block = 2;

  // Number of null vectors.
  // We're just doing the free laplace, so we only need one.
  const int coarse_dof = 1;

  // How many times to refine. 
  const int n_refine = 6; // (64 -> 32 -> 16 -> 8 -> 4 -> 2)

  // Information about the solve.

  // Solver tolerance.
  const double tol = 1e-8; 

  // Maximum iterations.
  const int max_iter = 1000;

  // Restart frequency
  const int restart_freq = 32;

  // Somewhere to solve inversion info.
  inversion_info invif;

  // Verbosity.
  inversion_verbose_struct verb;
  verb.verbosity = VERB_DETAIL;
  verb.verb_prefix = "Level 0: ";
  verb.precond_verbosity = VERB_NONE; //VERB_DETAIL;
  verb.precond_verb_prefix = "Prec ";

  // Create a lattice object for the fine lattice.
  Lattice2D** lats = new Lattice2D*[n_refine+1];
  lats[0] = new Lattice2D(x_len, y_len, dof);

  // Create a unit gauged laplace stencil.
  complex<double>* unit_gauge = allocate_vector<complex<double>>(lats[0]->get_size_gauge());
  unit_gauge_u1(unit_gauge, lats[0]);
  // Need to change relaxation params for gauged laplace.
  //read_gauge_u1(unit_gauge, lats[0], "../common_cfgs_u1/l64t64b60_heatbath.dat");
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
    // Arg 3: Should we construct the coarse stencil?
    // Arg 4: Does the operator have a sense of chirality?
    // Arg 4: What should we construct the coarse stencil from? (Not relevant yet.)
    // Arg 5: Non-block-orthogonalized null vector.
    mg_object->push_level(lats[i], transfer_objs[i-1], true, false, MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL, &null_vector);

    // Clean up local vector, since they get copied in.
    deallocate_vector(&null_vector);
  }

  // Create a StatefulMultigridMG. The multigrid solver uses this because
  // it has some extra functions to track a recursive MG solve.
  StatefulMultigridMG* stateful_mg_object = new StatefulMultigridMG(mg_object);

  // Use the same solver info on each level.
  const int n_pre_smooth = 2;
  const double pre_smooth_tol = 1e-15; // never
  const int n_post_smooth = 2;
  const double post_smooth_tol = 1e-15; // never
  const int coarse_max_iter = 1000000; // never
  const double coarse_tol = 0.2;
  const int coarse_restart = 32;

  // Create the same solver struct on each level.
  StatefulMultigridMG::LevelInfoMG** level_info_objs = new StatefulMultigridMG::LevelInfoMG*[n_refine];
  for (i = 0; i < n_refine; i++)
  {
    level_info_objs[i] = new StatefulMultigridMG::LevelInfoMG();
    level_info_objs[i]->pre_tol = pre_smooth_tol;
    level_info_objs[i]->pre_iters = n_pre_smooth;
    level_info_objs[i]->post_tol = post_smooth_tol;
    level_info_objs[i]->post_iters = n_post_smooth;
    level_info_objs[i]->coarse_tol = coarse_tol;
    level_info_objs[i]->coarse_iters = coarse_max_iter;
    level_info_objs[i]->coarse_restart_freq = coarse_restart; 
    // FILL IN VALUES
    stateful_mg_object->set_level_info(i, level_info_objs[i]);
  }

  // Prepare storage and a guess right hand side.

  // Create a right hand side, fill with gaussian random numbers.
  //complex<double>* b = allocate_vector<complex<double>>(lats[0]->get_size_cv());
  complex<double>* b = mg_object->check_out(0);
  gaussian(b, lats[0]->get_size_cv(), generator);
  double bnorm = sqrt(norm2sq(b, lats[0]->get_size_cv()));

  // Create a place to accumulate a solution. Assume a zero initial guess.
  complex<double>* x = mg_object->check_out(0);
  zero_vector(x, lats[0]->get_size_cv());

  // Create a place to compute Ax. Since we have a zero initial guess, this
  // starts as zero.
  complex<double>* Ax = mg_object->check_out(0);
  zero_vector(Ax, lats[0]->get_size_cv());

  ///////////////////
  // Non-MG solve! //
  ///////////////////
  /*invif = minv_vector_gcr_restart(x, b, lats[0]->get_size_cv(),
              max_iter, tol, restart_freq,
              apply_stencil_2D_M, (void*)mg_object->get_stencil(0));

  cout << "Simple GCR solve " << (invif.success ? "converged" : "failed to converge")
          << " in " << invif.iter << " iterations with alleged tolerance "
          << sqrt(invif.resSq)/bnorm << ".\n";
  zero_vector(Ax, lats[0]->get_size_cv());
  mg_object->apply_stencil(Ax, x, 0);
  cout << "Check tolerance " << sqrt(diffnorm2sq(b, Ax, lats[0]->get_size_cv()))/bnorm << "\n";
  */

  ////////////////////
  // K-cycle solve! //
  ////////////////////

  // Reset values.
  zero_vector(x, lats[0]->get_size_cv());
  zero_vector(Ax, lats[0]->get_size_cv());

  // Run a VPGCR solve!
  invif = minv_vector_gcr_var_precond_restart(x, b, lats[0]->get_size_cv(),
              max_iter, tol, restart_freq,
              apply_stencil_2D_M, (void*)mg_object->get_stencil(0),
              StatefulMultigridMG::mg_preconditioner, (void*)stateful_mg_object, &verb); 

  cout << "Multigrid " << (invif.success ? "converged" : "failed to converge")
          << " in " << invif.iter << " iterations with alleged tolerance "
          << sqrt(invif.resSq)/bnorm << ".\n";

  // Check solution.
  zero_vector(Ax, lats[0]->get_size_cv());
  mg_object->apply_stencil(Ax, x, 0);
  cout << "Check tolerance " << sqrt(diffnorm2sq(b, Ax, lats[0]->get_size_cv()))/bnorm << "\n";

  // Check vectors back in.
  mg_object->check_in(Ax, 0);
  mg_object->check_in(x, 0);
  mg_object->check_in(b, 0);

  ///////////////
  // Clean up. //
  ///////////////

  deallocate_vector(&unit_gauge);

  // Delete level info objects.
  for (i = 0; i < n_refine; i++)
  {
    delete level_info_objs[i];
  }
  delete[] level_info_objs;

  // Delete StatefulMultigridMG.
  delete stateful_mg_object;

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

