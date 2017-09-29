// Copyright (c) 2017 Evan S Weinberg
// A test of a K-cycle for the interacting
// Wilson operator with red-black and schur used
// at each level. Generates null vectors using
// BiCGstab-L.

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
#include "inverters/generic_bicgstab_l.h"

// QMG
#include "lattice/lattice.h"
#include "transfer/transfer.h"
#include "stencil/stencil_2d.h"
#include "multigrid/stateful_multigrid.h"

// Grab Wilson operator (we just use unit gauge fields -> free laplace)
#include "operators/wilson.h"
#include "u1/u1_utils.h"



// Forward declare mg preconditioner using dwf operator.
void dwf_mg_preconditioner(complex<double>* lhs, complex<double>* rhs, int size, void* extra_data, inversion_verbose_struct* verb);


int main(int argc, char** argv)
{
  // Iterators.
  int i,j,k;

  // Are we testing the free (two exact null vectors) or
  // interacting (four algebraic null vectors) case?
  const bool do_free = false;

  // Set output precision to be long.
  cout << setprecision(20);

  // Random number generator.
  std::mt19937 generator (1337u);

  // Basic information for fine level.
  const int x_len = 128;
  const int y_len = 128;
  const int dof = Wilson2D::get_dof();

  // Information on the Wilson operator.
  double mass;
  if (do_free)
  {
    mass = 0.1;
  }
  else
  {
    // Staggered specific information.
    // For 64^2, beta = 6.0, eigenvalues go negative around -0.075.
    mass = -0.07;
  }

  // Blocking size.
  const int x_block = 4;
  const int y_block = 4;

  // Number of null vectors.
  int coarse_dof;
  if (do_free)
  {
    coarse_dof = 2;
  }
  else
  {
    coarse_dof = 8;
  }

  // How many times to refine. 
  const int n_refine = 3; // (64 -> 16 -> 4 -> 1)

  // Information about the outermost solve.
  const double tol = 1e-8; 
  const int max_iter = 1000;
  const int restart_freq = 32;

  // Information about intermediate solves.
  const double inner_tol = 0.2;
  const int inner_max_iter = 1000;
  const int inner_restart_freq = 32;

  // Information about pre- and post-smooths. 
  const int n_pre_smooth = 2;
  const double pre_smooth_tol = 1e-15; // never
  const int n_post_smooth = 2;
  const double post_smooth_tol = 1e-15; // never

  // Information about the coarsest solve.
  const double coarsest_tol = 0.2;
  const int coarsest_max_iter = 1000;
  const int coarsest_restart_freq = 32;

  // What solve are we doing on each level?
  QMGStencilType solve_type = QMG_MATVEC_RIGHT_SCHUR;

  // Somewhere to solve inversion info.
  inversion_info invif;

  // Verbosity.
  inversion_verbose_struct verb;
  verb.verbosity = VERB_DETAIL;
  verb.verb_prefix = "Level 0: ";
  verb.precond_verbosity = VERB_DETAIL;
  verb.precond_verb_prefix = "Prec ";

  // Create a lattice object for the fine lattice.
  Lattice2D** lats = new Lattice2D*[n_refine+1];
  lats[0] = new Lattice2D(x_len, y_len, dof);

  // Prepare the gauge field.
  Lattice2D* lat_gauge = new Lattice2D(x_len, y_len, 1); // hack...
  complex<double>* gauge_field = allocate_vector<complex<double>>(lats[0]->get_size_gauge());
  if (do_free)
  {
    unit_gauge_u1(gauge_field, lat_gauge);
  }
  else
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
      default:
        std::cout << "[QMG-ERROR]: Invalid lattice size.\n";
        return -1;
        break;
    }
  }
  delete lat_gauge;

  // Create a Wilson stencil.
  Wilson2D* wilson_op = new Wilson2D(lats[0], mass, gauge_field);

  // Prepare for rbjacobi solve.
  wilson_op->build_rbjacobi_stencil();

  // Prepare level solve objects for the top level.
  StatefulMultigridMG::LevelSolveMG** level_solve_objs = new StatefulMultigridMG::LevelSolveMG*[n_refine];

  // Prepare coarsest solve object for the coarsest level.
  StatefulMultigridMG::CoarsestSolveMG* coarsest_solve_obj = new StatefulMultigridMG::CoarsestSolveMG;
  coarsest_solve_obj->coarsest_stencil_app = solve_type;
  coarsest_solve_obj->coarsest_tol = coarsest_tol;
  coarsest_solve_obj->coarsest_iters = coarsest_max_iter;
  coarsest_solve_obj->coarsest_restart_freq = coarsest_restart_freq;

  // Create a MultigridMG object, push top level onto it!
  StatefulMultigridMG* mg_object = new StatefulMultigridMG(lats[0], wilson_op, coarsest_solve_obj);

  // What type of stencil should we coarsen?
  MultigridMG::QMGMultigridPrecondStencil stencil_to_coarsen = MultigridMG::QMG_MULTIGRID_PRECOND_RIGHT_BLOCK_JACOBI;
  //MultigridMG::QMGMultigridPrecondStencil stencil_to_coarsen = MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL;

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

    // Create a new null vectors. These are copied into local memory in the
    // transfer object, so we can create and destroy these in this loop.
    complex<double>** null_vectors = new complex<double>*[coarse_dof];
    if (do_free)
    {
      // Two null vectors, one for each parity component.

      // Top.
      null_vectors[0] = allocate_vector<complex<double>>(lats[i-1]->get_size_cv());
      zero_vector(null_vectors[0], lats[i-1]->get_size_cv());
      constant_vector_blas(null_vectors[0], 2, 1.0, lats[i-1]->get_size_cv()/2);

      // Bottom.
      null_vectors[1] = allocate_vector<complex<double>>(lats[i-1]->get_size_cv());
      zero_vector(null_vectors[1], lats[i-1]->get_size_cv());
      constant_vector_blas(null_vectors[1]+1, 2, 1.0, lats[i-1]->get_size_cv()/2);
    }
    else
    {
      // Create coarse_dof null vectors.
      for (j = 0; j < coarse_dof/2; j++)
      {
        // Update verbosity string.
        verb.verb_prefix = "Level " + to_string(i) + " Null Vector " + to_string(j) + " ";

        // Will become up chiral projection
        null_vectors[j] = allocate_vector<complex<double> >(lats[i-1]->get_size_cv());
        zero_vector(null_vectors[j], lats[i-1]->get_size_cv());

        // Check out vector.
        complex<double>* rand_guess = mg_object->get_storage(i-1)->check_out();

        // Fill with random numbers.
        gaussian(rand_guess, lats[i-1]->get_size_cv(), generator);

        // Make orthogonal to previous vectors.
        for (k = 0; k < j; k++)
          orthogonal(rand_guess, null_vectors[k], lats[i-1]->get_size_cv());

        // Check out vector for residual equation.
        complex<double>* Arand_guess = mg_object->get_storage(i-1)->check_out();

        // Zero, form residual.
        zero_vector(Arand_guess, lats[i-1]->get_size_cv());
        mg_object->get_stencil(i-1)->apply_M(Arand_guess, rand_guess, QMG_MATVEC_RIGHT_JACOBI);
        cax(-1.0, Arand_guess, lats[i-1]->get_size_cv());

        // Solve residual equation.
        //minv_vector_bicgstab_l(null_vectors[j], Arand_guess, lats[i-1]->get_size_cv(), 500, 5e-5, 6, apply_stencil_2D_M_rbjacobi, (void*)mg_object->get_stencil(i-1), &verb);
        minv_vector_gcr_restart(null_vectors[j], Arand_guess, lats[i-1]->get_size_cv(), 500, 5e-5, 64, apply_stencil_2D_M_rbjacobi, (void*)mg_object->get_stencil(i-1), &verb);

        // Undo residual equation.
        cxpy(rand_guess, null_vectors[j], lats[i-1]->get_size_cv());

        // Check in.
        mg_object->get_storage(i-1)->check_in(rand_guess);
        mg_object->get_storage(i-1)->check_in(Arand_guess);

        // Orthogonalize against previous vectors.
        for (k = 0; k < j; k++)
          orthogonal(null_vectors[j], null_vectors[k], lats[i-1]->get_size_cv());
      }

      // Perform chiral projection.
      for (j = 0; j < coarse_dof/2; j++)
      {
        // Get new vector.
        null_vectors[j+lats[i]->get_nc()/2] = allocate_vector<complex<double> >(lats[i-1]->get_size_cv());

        // Perform chiral projection, putting the "down" projection into the second
        // vector and keeping the "up" projection in the first vector.
        mg_object->get_stencil(i-1)->chiral_projection_both(null_vectors[j], null_vectors[j+lats[i]->get_nc()/2]);

        // Normalize.
        normalize(null_vectors[j], lats[i-1]->get_size_cv());
        normalize(null_vectors[j+lats[i]->get_nc()/2], lats[i-1]->get_size_cv());
      }
    }

    // Create and populate a transfer object.
    // Fine lattice, coarse lattice, null vector(s), perform the block ortho, preserve cholesky, how did we double
    transfer_objs[i-1] = new TransferMG(lats[i-1], lats[i], null_vectors, true, false, QMG_DOUBLE_PROJECTION);

    // Prepare a new LevelSolveMG object for the new level.
    level_solve_objs[i-1] = new StatefulMultigridMG::LevelSolveMG;
    level_solve_objs[i-1]->fine_stencil_app = solve_type;
    level_solve_objs[i-1]->intermediate_tol = inner_tol;
    level_solve_objs[i-1]->intermediate_iters = inner_max_iter;
    level_solve_objs[i-1]->intermediate_restart_freq = inner_restart_freq;
    level_solve_objs[i-1]->pre_tol = pre_smooth_tol;
    level_solve_objs[i-1]->pre_iters = n_pre_smooth;
    level_solve_objs[i-1]->post_tol = post_smooth_tol;
    level_solve_objs[i-1]->post_iters = n_post_smooth;

    // Push a new level on the multigrid object! Also, save the global null vector.
    // Arg 1: New lattice
    // Arg 2: New transfer object (between new and prev lattice)
    // Arg 3: Should we construct the coarse stencil?
    // Arg 4: Is the operator chiral? (True for Wilson)
    // Arg 5: What should we construct the coarse stencil from?
    // Arg 6: Should we prep dagger or rbjacobi stencil (rbjacobi, for this test)
    // Arg 7: Non-block-orthogonalized null vector.
    mg_object->push_level(lats[i], transfer_objs[i-1], level_solve_objs[i-1], true, Wilson2D::has_chirality(), stencil_to_coarsen, CoarseOperator2D::QMG_COARSE_BUILD_RBJACOBI, null_vectors);

    // Clean up local vector, since they get copied in.
    for (j = 0; j < coarse_dof; j++)
      deallocate_vector(&null_vectors[j]);
    delete[] null_vectors;
  }

  // What type of solve are we doing?
  matrix_op_cplx apply_stencil_op = Stencil2D::get_apply_function(solve_type);
  int solve_size = (solve_type == QMG_MATVEC_RIGHT_SCHUR ? lats[0]->get_size_cv()/2 : lats[0]->get_size_cv());

  // Do a sanity check.
  if (stencil_to_coarsen == MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL)
  {
    if (solve_type != QMG_MATVEC_ORIGINAL)
    {
      std::cout << "[QMG-ERROR]: Cannot do right jacobi solve without building coarse rbjacobi stencils.\n";
      return 0;
    }
  }
  else // QMG_MULTIGRID_PRECOND_RIGHT_BLOCK_JACOBI
  {
    if (solve_type == QMG_MATVEC_ORIGINAL)
    {
      std::cout << "[QMG-ERROR]: Cannot do un-preconditioned solve with the rbjacobi stencils.\n";
      return 0;
    }
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

  // Prepare b.
  complex<double>* b_prep = mg_object->check_out(0);
  zero_vector(b_prep, lats[0]->get_size_cv());
  mg_object->get_stencil(0)->prepare_M(b_prep, b, solve_type);

  // Run a VPGCR solve!
  invif = minv_vector_gcr_var_precond_restart(x, b_prep, solve_size,
              max_iter, tol, restart_freq,
              apply_stencil_op, (void*)mg_object->get_stencil(0),
              StatefulMultigridMG::mg_preconditioner, (void*)mg_object, &verb); 

  cout << "Multigrid " << (invif.success ? "converged" : "failed to converge")
          << " in " << invif.iter << " iterations with alleged tolerance "
          << sqrt(invif.resSq)/bnorm << ".\n";

  // Reconstruct x.
  complex<double>* x_reconstruct = mg_object->check_out(0);
  zero_vector(x_reconstruct, lats[0]->get_size_cv());
  mg_object->get_stencil(0)->reconstruct_M(x_reconstruct, x, b, solve_type);

  // Check solution.
  zero_vector(Ax, lats[0]->get_size_cv());
  mg_object->apply_stencil(Ax, x_reconstruct, 0);
  cout << "Check tolerance " << sqrt(diffnorm2sq(b, Ax, lats[0]->get_size_cv()))/bnorm << "\n";

  // Check vectors back in.
  mg_object->check_in(b_prep, 0);
  mg_object->check_in(x_reconstruct, 0);
  mg_object->check_in(Ax, 0);
  mg_object->check_in(x, 0);
  mg_object->check_in(b, 0);

  ///////////////
  // Clean up. //
  ///////////////

  deallocate_vector(&gauge_field);

  // Delete MultigridMG.
  delete mg_object;

  // Delete transfer objects.
  for (i = 0; i < n_refine; i++)
  {
    delete transfer_objs[i];
  }
  delete[] transfer_objs;

  // Delete stencil.
  delete wilson_op;

  // Delete coarsest solve objects.
  delete coarsest_solve_obj;

  // Delete level solve objects.
  for (i = 0; i < n_refine; i++)
  {
    delete level_solve_objs[i];
  }
  delete[] level_solve_objs;

  // Delete lattices.
  for (i = 0; i <= n_refine; i++)
  {
    delete lats[i];
  }
  delete[] lats; 

  return 0;
}

