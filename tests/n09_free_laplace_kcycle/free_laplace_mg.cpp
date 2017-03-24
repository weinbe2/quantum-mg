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
    if (level > 0)
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
    if (level < mg_object->get_num_levels()-1)
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
}

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
  const double mass = 0.01;

  // Blocking size.
  const int x_block = 2;
  const int y_block = 2;

  // Number of null vectors.
  // We're just doing the free laplace, so we only need one.
  const int coarse_dof = 1;

  // How many times to refine. 
  const int n_refine = 5; // (64 -> 32 -> 16 -> 8 -> 4 -> 2)

  // Information about the solve.

  // Solver tolerance.
  const double tol = 1e-8; 

  // Maximum iterations.
  const int max_iter = 1000;

  // Restart frequency
  const int restart_freq = 32;

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
  StatefulMultigridMG stateful_mg_object = new StatefulMultigridMG(mg_object);

  ////////////////////
  // K-cycle solve! //
  ////////////////////

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

  // Create a place to store the residual. Since we have zero initial guess,
  // the initial residual is b - Ax = b.
  complex<double>* r = mg_object->check_out(0);
  copy_vector(r, b, lats[0]->get_size_cv());

  // Create a place to store the current residual norm.
  double rnorm; 

   // Create a place to store the error.
  complex<double>* e = mg_object->check_out(0);
  zero_vector(e, lats[0]->get_size_cv());

  // Run a VPGCR solve!
  minv_vector_gcr_var_precond_restart(x, b, lats[0]->get_size_cv(),
              max_iter, tol, restart_freq,
              apply_stencil_2D_M, (void*)&mg_object->get_stencil(i),
              mg_preconditioner, (void*)&stateful_mg_object); //, &verb); 

  apply_stencil_2D_M(x, b, void* extra_data)

  for (i = 0; i < max_iter; i++)
  {
    // Zero the error.
    zero_vector(e, lats[0]->get_size_cv());

    // Enter a v-cycle.
    richardson_vcycle(e, r, omega_refine, n_relax, mg_object, 0);

    // Update the solution.
    cxpy(e, x, lats[0]->get_size_cv());

    // Update the residual.
    zero_vector(Ae, lats[0]->get_size_cv());
    mg_object->apply_stencil(Ae, e, 0); // top level.
    caxpy(-1.0, Ae, r, lats[0]->get_size_cv());

    // Check norm.
    rnorm = sqrt(norm2sq(r, lats[0]->get_size_cv()));
    cout << "Full V-cycle Outer step " << i << ": tolerance " << rnorm/bnorm << "\n";
    if (rnorm/bnorm < tol)
      break; 
  }

  // Check solution.
  zero_vector(Ax, lats[0]->get_size_cv());
  mg_object->apply_stencil(Ax, x, 0);
  cout << "Check tolerance " << sqrt(diffnorm2sq(b, Ax, lats[0]->get_size_cv()))/bnorm << "\n";

  // Check vectors back in.
  mg_object->check_in(e, 0);
  mg_object->check_in(r, 0);
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

// Perform nrich Richardson iterations at a given level.
// Solves A e = r, assuming e is zeroed, using Ae as temporary space.
void richardson_kernel(complex<double>* e, complex<double>* r,
                      vector<double>& omega, int nrich, MultigridMG* mg_obj, int level)
{
  // Simple check.
  if (nrich <= 0)
    return;

  // Get vector size.
  const int vec_size = mg_obj->get_lattice(level)->get_size_cv();

  // Remember, this routine assumes e is zero.
  // This means the first iter doesn't do anything. 
  // Just update e += omega(r - Ax) -> e += omega r.
  caxy(omega[level], r, e, vec_size);

  if (nrich == 1)
    return; 

  // Relax on the residual via Richardson. (Looks like pre-smoothing.)
  // e = A^{-1} r, via the remaining nrich-1 iterations.
  complex<double>* Ae = mg_obj->check_out(level);

  for (int i = 1; i < nrich; i++)
  {
    zero_vector(Ae, vec_size);
    mg_obj->apply_stencil(Ae, e, level); // top level stencil.

    // e += omega(r - Ax)
    caxpbypz(omega[level], r, -omega[level], Ae, e, vec_size);
  }

  mg_obj->check_in(Ae, level);

}

// Perform one iteraiton of a V cycle using the richardson kernel.
// Acts recursively. 
void richardson_vcycle(complex<double>* e, complex<double>* r, vector<double>& omega,
                    int nrich, MultigridMG* mg_obj, int level)
{
  const int fine_size = mg_obj->get_lattice(level)->get_size_cv();

  // If we're at the bottom level, just smooth and send it back up.
  if (level == mg_obj->get_num_levels()-1)
  {
    // Zero out the error.
    zero_vector<complex<double>>(e, fine_size);

    // Kernel it up.
    richardson_kernel(e, r, omega, nrich, mg_obj, level);
  }
  else // all aboard the V-cycle traiiiiiiiin.
  {
    const int coarse_size = mg_obj->get_lattice(level+1)->get_size_cv();

    // We need temporary vectors everywhere for mat-vecs. Grab that here.
    complex<double>* Atmp = mg_obj->check_out(level);

    // First stop: presmooth. Solve A z1 = r, form new residual r1 = r - Az1.
    complex<double>* z1 = mg_obj->check_out(level);
    zero_vector(z1, fine_size);
    richardson_kernel(z1, r, omega, nrich, mg_obj, level);
    zero_vector(Atmp, fine_size);
    mg_obj->apply_stencil(Atmp, z1, level);
    complex<double>* r1 = mg_obj->check_out(level);
    caxpbyz(1.0, r, -1.0, Atmp, r1, fine_size);

    // Next stop! Restrict, recurse, prolong, etc.
    complex<double>* r_coarse = mg_obj->check_out(level+1);
    zero_vector(r_coarse, coarse_size);
    mg_obj->restrict_f2c(r1, r_coarse, level);
    mg_obj->check_in(r1, level);
    complex<double>* e_coarse = mg_obj->check_out(level+1);
    zero_vector(e_coarse, coarse_size);
    richardson_vcycle(e_coarse, r_coarse, omega, nrich, mg_obj, level+1);
    mg_obj->check_in(r_coarse, level+1);
    complex<double>* z2 = mg_obj->check_out(level);
    zero_vector(z2, fine_size);
    mg_obj->prolong_c2f(e_coarse, z2, level);
    mg_obj->check_in(e_coarse, level+1);
    zero_vector(e, fine_size);
    cxpyz(z1, z2, e, fine_size);
    mg_obj->check_in(z1, level);
    mg_obj->check_in(z2, level);

    // Last stop, post smooth. Form r2 = r - A(z1 + z2) = r - Ae, solve A z3 = r2
    zero_vector(Atmp, fine_size);
    mg_obj->apply_stencil(Atmp, e, level);
    complex<double>* r2 = mg_obj->check_out(level);
    caxpbyz(1.0, r, -1.0, Atmp, r2, fine_size); 
    complex<double>* z3 = mg_obj->check_out(level);
    zero_vector(z3, fine_size);
    richardson_kernel(z3, r2, omega, nrich, mg_obj, level);
    cxpy(z3, e, fine_size);

    // We're done with Atmp (vector 2), r2 (vector 3), z3 (vector 4)
    mg_obj->check_in(Atmp, level);
    mg_obj->check_in(r2, level);
    mg_obj->check_in(z3, level);

    // And we're (theoretically) done!
  }
}