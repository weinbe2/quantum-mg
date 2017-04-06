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
#include "multigrid/multigrid.h"

// Grab laplace operator (we just use unit gauge fields -> free laplace)
#include "operators/gaugedlaplace.h"
#include "u1/u1_utils.h"

// Maybe we want to put this directly into MultigridMG, but for now,
// this is a wrapper struct that contains MultigridMG as well
// as the current level.
class StatefulMultigridMG
{
private:
  // Get rid of copy, assignment.
  StatefulMultigridMG(StatefulMultigridMG const &);
  StatefulMultigridMG& operator=(StatefulMultigridMG const &);

  // Internal variables.
  MultigridMG* mg_object;
  int current_level;

public:

  // Simple constructor.
  StatefulMultigridMG(MultigridMG* mg_object, int current_level = 0)
    : mg_object(mg_object), current_level(current_level) { ; }

  // Get multigrid object
  MultigridMG* get_multigrid_object()
  {
    return mg_object;
  }

  // Set the multigrid level.
  void set_multigrid_level(int level)
  {
    if (level >= 0 && level < mg_object->get_num_levels())
    {
      current_level = level;
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: StatefulMultigridMG->current_level " << level << " is outside of [0,max_level-1].\n";
    }
  }

  // Go one level finer.
  void go_finer()
  {
    if (current_level > 0)
    {
      current_level--;
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot go finer than the top level in StatefulMultigridMG.\n";
    }
  }

  // Go one level coarser.
  void go_coarser()
  {
    if (current_level < mg_object->get_num_levels()-2)
    {
      current_level++;
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: Cannot go coarser than the second-coarsest level in StatefulMultigridMG.\n";
    }
  }

  // Get the level.
  int get_multigrid_level()
  {
    return current_level;
  }
};

void mg_preconditioner(complex<double>* lhs, complex<double>* rhs, int size, void* extra_data, inversion_verbose_struct* verb = 0);

// Perform nrich Richardson iterations at a given level.
// Solves A e = r, assuming e is zeroed.
void richardson_kernel(complex<double>* e, complex<double>* r, vector<double>& omega,
                    int nrich, MultigridMG* mg_obj, int level);

// Perform one iteraiton of a V cycle using the richardson kernel.
// Acts recursively. 
void richardson_vcycle(complex<double>* e, complex<double>* r, vector<double>& omega,
                    int nrich, MultigridMG* mg_obj, int level);

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
  const int dof = 1; 

  // Information on the Laplace operator.
  const double mass = 0.001;

  // Blocking size.
  const int x_block = 2;
  const int y_block = 2;

  // Number of null vectors.
  // We're just doing the free laplace, so we only need one.
  const int coarse_dof = 1;

  // How many times to refine. 
  const int n_refine = 7; // (64 -> 32 -> 16 -> 8 -> 4 -> 2)

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
    // Arg 4: What should we construct the coarse stencil from? (Not relevant yet.)
    // Arg 5: Non-block-orthogonalized null vector.
    mg_object->push_level(lats[i], transfer_objs[i-1], true, MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL, &null_vector);

    // Clean up local vector, since they get copied in.
    deallocate_vector(&null_vector);
  }

  // Create a StatefulMultigridMG. The multigrid solver uses this because
  // it has some extra functions to track a recursive MG solve.
  StatefulMultigridMG* stateful_mg_object = new StatefulMultigridMG(mg_object);

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
              mg_preconditioner, (void*)stateful_mg_object, &verb); 

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

void mg_preconditioner(complex<double>* lhs, complex<double>* rhs, int size, void* extra_data, inversion_verbose_struct* verb)
{
  // Expose the Multigrid objects.

  // State.
  StatefulMultigridMG* stateful_mg_object = (StatefulMultigridMG*)extra_data;
  int level = stateful_mg_object->get_multigrid_level();
  cout << "Entered level " << level << "\n" << flush;

  // MultigridMG.
  MultigridMG* mg_object = stateful_mg_object->get_multigrid_object();

  // Stencils.
  Stencil2D* fine_stencil = mg_object->get_stencil(level);
  Stencil2D* coarse_stencil = mg_object->get_stencil(level+1);

  // Transfer object.
  TransferMG* transfer = mg_object->get_transfer(level);

  // Storage objects.
  ArrayStorageMG<complex<double>>* fine_storage = mg_object->get_storage(level);
  ArrayStorageMG<complex<double>>* coarse_storage = mg_object->get_storage(level+1);

  // Sizes.
  int fine_size = mg_object->get_lattice(level)->get_size_cv();
  int coarse_size = mg_object->get_lattice(level+1)->get_size_cv();

  // Number of levels.
  int total_num_levels = mg_object->get_num_levels();

  // Verbosity and inversion info.
  inversion_info invif;
  inversion_verbose_struct verb2;
  verb2.verbosity = VERB_NONE;
  verb2.verb_prefix = "Level " + to_string(level+1) + ": ";
  verb2.precond_verbosity = VERB_NONE; //VERB_DETAIL;
  verb2.precond_verb_prefix = "Prec ";

  // Hard code various things for now.
  int n_pre_smooth = 2;
  double pre_smooth_tol = 1e-15; // never
  int n_post_smooth = 2;
  double post_smooth_tol = 1e-15; // never
  int coarse_max_iter = 1000000; // never
  double coarse_tol = 0.2;
  int coarse_restart = 32;

  // We need a temporary vector for mat-vecs everywhere.
  complex<double>* Atmp = fine_storage->check_out();

  // Step 1: presmooth.
  // Solve A z1 = rhs, form new residual r1 = rhs - A z1
  complex<double>* z1 = fine_storage->check_out();
  zero_vector(z1, fine_size);
  minv_vector_gcr_restart(z1, rhs, fine_size, n_pre_smooth, pre_smooth_tol, coarse_restart, apply_stencil_2D_M, (void*)fine_stencil);
  zero_vector(Atmp, fine_size);
  fine_stencil->apply_M(Atmp, z1);
  complex<double>* r1 = fine_storage->check_out();
  caxpbyz(1.0, rhs, -1.0, Atmp, r1, fine_size);

  // Next stop: restrict, recurse (or coarsest solve), prolong.
  complex<double>* r_coarse = coarse_storage->check_out();
  zero_vector(r_coarse, coarse_size);
  transfer->restrict_f2c(r1, r_coarse);
  fine_storage->check_in(r1);
  complex<double>* e_coarse = coarse_storage->check_out();
  zero_vector(e_coarse, coarse_size);
  if (level == total_num_levels-2) // if we're already on the coarsest level
  {
    // Do coarsest solve.
    verb2.verbosity = VERB_NONE;
    invif = minv_vector_gcr_restart(e_coarse, r_coarse, coarse_size,
                        coarse_max_iter, coarse_tol, coarse_restart, 
                        apply_stencil_2D_M, (void*)coarse_stencil, &verb2);
  }
  else
  {
    // Recurse.
    stateful_mg_object->go_coarser();
    // K cycle
    invif = minv_vector_gcr_var_precond_restart(e_coarse, r_coarse, coarse_size,
                        coarse_max_iter, coarse_tol, coarse_restart,
                        apply_stencil_2D_M, (void*)coarse_stencil,
                        mg_preconditioner, (void*)stateful_mg_object, &verb2);
    // V cycle
    //mg_preconditioner(e_coarse, r_coarse, coarse_size, (void*)stateful_mg_object);
    stateful_mg_object->go_finer();
  }
  cout << "Level " << level << " coarse preconditioner took " << invif.iter << " iterations.\n" << flush;
  coarse_storage->check_in(r_coarse);
  complex<double>* z2 = fine_storage->check_out();
  zero_vector(z2, fine_size);
  transfer->prolong_c2f(e_coarse, z2);
  coarse_storage->check_in(e_coarse);
  zero_vector(lhs, fine_size);
  cxpyz(z1, z2, lhs, fine_size);
  fine_storage->check_in(z1);
  fine_storage->check_in(z2);

  // Last stop, post smooth. Form r2 = r - A(z1 + z2) = r - Ae, solve A z3 = r2.
  zero_vector(Atmp, fine_size);
  fine_stencil->apply_M(Atmp, lhs);
  complex<double>* r2 = fine_storage->check_out();
  caxpbyz(1.0, rhs, -1.0, Atmp, r2, fine_size);
  complex<double>* z3 = fine_storage->check_out();
  zero_vector(z3, fine_size);
  minv_vector_gcr(z3, r2, fine_size, n_post_smooth, post_smooth_tol, apply_stencil_2D_M, (void*)fine_stencil);
  cxpy(z3, lhs, fine_size);

  // Check vectors back in.
  fine_storage->check_in(Atmp);
  fine_storage->check_in(r2);
  fine_storage->check_in(z3);

  cout << "Exited level " << level << "\n" << flush;
}
