// Copyright (c) 2017 Evan S Weinberg
// A test of an algebraic V-cycle for the free laplace equation.
// Just uses relaxation, so not a K-cycle!

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <random>
#include <vector>

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

// I think we need a "stateful" MG object. This needs some design...
// In principle, it could have pre-allocated storage, know about
// what to do at each level: number of presmooths, preconditioning,
// postsmooths, etc. For now, this just has storage. 
// Right now, this is a struct of arrays. An array of structs would be
// better: one for each level. The MultigridMG object could hold it.
class MultigridStorage
{
private:
  // Get rid of copy, assignment operator.
  MultigridStorage(MultigridStorage const &);
  MultigridStorage& operator=(MultigridStorage const &);

  // First index: level.
  // Second index: which vector.
  vector< vector< complex<double>* > > vector_storage;

  // Holds onto lattices for state.
  Lattice2D** lats;

public:

  // Allocation constructor. Currently just specifies number of levels,
  // how many arrays to allocate per level (should be a vector listing
  // how many at each level, but eh).
  // Eventually, this will take in nlevels and a lattice object, and we'd
  // call it a day.
  MultigridStorage(Lattice2D** lats, int nlevels, int nvecs_per_level)
    : lats(lats)
  {
    for (int i = 0; i < nlevels; i++)
    {
      vector< complex<double>* > vec_building;
      for (int j = 0; j < nvecs_per_level; j++)
      {
        vec_building.push_back(allocate_vector<complex<double>>(lats[i]->get_size_cv()));
      }
      vector_storage.push_back(vec_building);
    }
  }

  // Clean up.
  ~MultigridStorage()
  {
    // Deallocate memory.
    for (unsigned int i = 0; i < vector_storage.size(); i++)
    {
      for (unsigned int j = 0; j < vector_storage[i].size(); j++)
      {
        deallocate_vector(&vector_storage[i][j]);
      }
    }
  }

  // Grab a temp vector. Maybe we should have a way to 'check out'
  // and 'return' a vector? Have it give a vector and a key?
  // This way, memory only gets allocated when it needs to, we don't
  // have to prespecify an amount... hmmm.
  complex<double>* get_temp_vector(int level, int num)
  {
    return vector_storage[level][num];
  }

};

// Perform nrich Richardson iterations at a given level.
// Solves A e = r, assuming e is zeroed, using Ae as temporary space.
void richardson_kernel(complex<double>* e, complex<double>* r, complex<double>* Ae,
                    vector<double>& omega, int nrich, MultigridMG* mg_obj, int level);

// Perform one iteraiton of a V cycle using the richardson kernel.
// Acts recursively. 
void richardson_vcycle(complex<double>* e, complex<double>* r, vector<double>& omega,
                    int nrich, MultigridMG* mg_obj, MultigridStorage* mg_storage,
                    int level);

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
  const int n_refine = 6; // (64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)

  // Information about the solve.

  // Solver tolerance.
  const double tol = 1e-8; 

  // Maximum iterations.
  const int max_iter = 1000;

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

  // Build a MultigridStorage object. This just allocates static memory
  // for the recursive solve so we aren't allocating/deallocating space
  // as we recurse. Eventually the MultigridStorage object will go
  // into the MultigridMG class. 
  // Let's pretend we need 6 vectors, we'll fix that later. 
  MultigridStorage* mg_storage = new MultigridStorage(lats, mg_object->get_num_levels(), 6);

  /////////////////////////////////////////////////////////////
  // Alright! Let's do a solve that isn't MG preconditioned. //
  /////////////////////////////////////////////////////////////

  // We're going to solve Ax = b with Richardson iterations.

  // Relaxation parameter---standard jacobi preconditioner
  // for nd-dimensional Laplace.
  const double relax_omega = 1.5/(4.0*lats[0]->get_nd()+mass*mass);

  // How many iters of relax to apply per outer loop.
  const int n_relax = 10; 

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

  /////////////////////////////////////////////////////////////////////
  // Be lazy and use power iterations to get the largest eigenvalue. //
  /////////////////////////////////////////////////////////////////////

  /*for (j = 0; j <= n_refine; j++)
  {
    complex<double>* piter = mg_storage->get_temp_vector(j, 0);
    complex<double>* Apiter = mg_storage->get_temp_vector(j, 1);
    gaussian(piter, lats[j]->get_size_cv(), generator);
    double nrm = sqrt(norm2sq(piter, lats[j]->get_size_cv()));
    cax(1.0/nrm, piter, lats[j]->get_size_cv());

    // do some large number of iterations
    for (i = 0; i < 10000; i++)
    {
      zero_vector(Apiter, lats[j]->get_size_cv());
      mg_object->apply_stencil(Apiter, piter, j);
      nrm = re_dot(piter, Apiter, lats[j]->get_size_cv());
      caxy(1.0/sqrt(norm2sq(Apiter, lats[j]->get_size_cv())), Apiter, piter, lats[j]->get_size_cv());
    }

    cout << "Largest eigenvalue level " << j << " approaches " << nrm << "\n";
  }*/

  ////////////////////////////////////////
  // Last bit: A fully recursive solve! //
  ////////////////////////////////////////

  // Set up the relaxation parameters at each level.
  // The maximum eigenvalue shrinks by a factor of blocksize
  // each refinement (except for the additive mass).
  vector<double> omega_refine;
  double factor = 1.0; // division factor.
  for (i = 0; i <= n_refine; i++)
  {
    if (lats[i]->get_volume() == 1)
    {
      omega_refine.push_back(1.0/(mass*mass)); // solve it exactly
    }
    else
    {
      omega_refine.push_back(1.33333333333333/(4.0*lats[0]->get_nd()*factor+mass*mass));
      factor /= ((double)x_block);
    }
  }


  // Grab fine (level 0) residuals, errors from storage 0 and 1.
  complex<double>* r_recursive = mg_storage->get_temp_vector(0, 0);
  complex<double>* e_recursive = mg_storage->get_temp_vector(0, 1);

  // Grab temporary space for Ae from storage 2.
  complex<double>* Ae_recursive = mg_storage->get_temp_vector(0, 2);

  // Clean up a bit.
  zero_vector(x, lats[0]->get_size_cv());
  copy_vector(r_recursive, b, lats[0]->get_size_cv());

  for (i = 0; i < max_iter; i++)
  {
    // Zero the error.
    zero_vector(e_recursive, lats[0]->get_size_cv());

    // Enter a v-cycle.
    richardson_vcycle(e_recursive, r_recursive, omega_refine, n_relax, mg_object, mg_storage, 0);

    // Update the solution.
    cxpy(e_recursive, x, lats[0]->get_size_cv());

    // Update the residual.
    zero_vector(Ae_recursive, lats[0]->get_size_cv());
    mg_object->apply_stencil(Ae_recursive, e_recursive, 0); // top level.
    caxpy(-1.0, Ae_recursive, r_recursive, lats[0]->get_size_cv());

    // Check norm.
    rnorm = sqrt(norm2sq(r_recursive, lats[0]->get_size_cv()));
    cout << "Full V-cycle Outer step " << i << ": tolerance " << rnorm/bnorm << "\n";
    if (rnorm/bnorm < tol)
      break; 
  }

  // Done with r_recursive, e_recursive, Ae_recursive. 

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

  deallocate_vector(&unit_gauge);

  // Delete MultigridStorage.
  delete mg_storage;

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
void richardson_kernel(complex<double>* e, complex<double>* r, complex<double>* Ae,
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
  for (int i = 1; i < nrich; i++)
  {
    zero_vector(Ae, vec_size);
    mg_obj->apply_stencil(Ae, e, level); // top level stencil.

    // e += omega(r - Ax)
    caxpbypz(omega[level], r, -omega[level], Ae, e, vec_size);
  }

}

// Perform one iteraiton of a V cycle using the richardson kernel.
// Acts recursively. 
void richardson_vcycle(complex<double>* e, complex<double>* r, vector<double>& omega,
                    int nrich, MultigridMG* mg_obj, MultigridStorage* mg_storage,
                    int level)
{
  const int fine_size = mg_obj->get_lattice(level)->get_size_cv();

  // If we're at the bottom level, just smooth and send it back up.
  if (level == mg_obj->get_num_levels()-1)
  {
    // Yeah, we need check-out storage... '0' is the coarsened residual,
    //                                    '1' is the coarsened error...
    complex<double>* Ae = mg_storage->get_temp_vector(level, 2);

    // Zero out the error.
    zero_vector<complex<double>>(e, fine_size);

    // Kernel it up.
    richardson_kernel(e, r, Ae, omega, nrich, mg_obj, level);
  }
  else // all aboard the V-cycle traiiiiiiiin.
  {
    const int coarse_size = mg_obj->get_lattice(level+1)->get_size_cv();

    // We need temporary vectors everywhere for mat-vecs. Grab that here.
    complex<double>* Atmp = mg_storage->get_temp_vector(level, 2);

    // First stop: presmooth. Solve A z1 = r, form new residual r1 = r - Az1.
    complex<double>* z1 = mg_storage->get_temp_vector(level, 3);
    zero_vector(z1, fine_size);
    richardson_kernel(z1, r, Atmp, omega, nrich, mg_obj, level);
    zero_vector(Atmp, fine_size);
    mg_obj->apply_stencil(Atmp, z1, level);
    complex<double>* r1 = mg_storage->get_temp_vector(level, 4);
    caxpbyz(1.0, r, -1.0, Atmp, r1, fine_size);

    // Next stop! Restrict, recurse, prolong, etc.
    complex<double>* r_coarse = mg_storage->get_temp_vector(level+1, 0);
    zero_vector(r_coarse, coarse_size);
    mg_obj->restrict_f2c(r1, r_coarse, level);
    // We're done with r1 (vector 4)
    complex<double>* e_coarse = mg_storage->get_temp_vector(level+1, 1);
    zero_vector(e_coarse, coarse_size);
    richardson_vcycle(e_coarse, r_coarse, omega, nrich, mg_obj, mg_storage, level+1);
    complex<double>* z2 = mg_storage->get_temp_vector(level, 4); 
    zero_vector(z2, fine_size);
    mg_obj->prolong_c2f(e_coarse, z2, level);
    zero_vector(e, fine_size);
    cxpyz(z1, z2, e, fine_size);
    // We're done with z1 (vector 3), z2 (vector 4)

    // Last stop, post smooth. Form r2 = r - A(z1 + z2) = r - Ae, solve A z3 = r2
    zero_vector(Atmp, fine_size);
    mg_obj->apply_stencil(Atmp, e, level);
    complex<double>* r2 = mg_storage->get_temp_vector(level, 3);
    caxpbyz(1.0, r, -1.0, Atmp, r2, fine_size); 
    complex<double>* z3 = mg_storage->get_temp_vector(level, 4);
    zero_vector(z3, fine_size);
    richardson_kernel(z3, r2, Atmp, omega, nrich, mg_obj, level);
    cxpy(z3, e, fine_size);

    // We're done with Atmp (vector 2), r2 (vector 3), z3 (vector 4)

    // And we're (theoretically) done!
  }
}