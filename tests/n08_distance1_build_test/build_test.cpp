// Copyright (c) 2017 Evan S Weinberg
// Test of building the coarse stencil 

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
  int i; 

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

  // Number of null vectors. Since we're doing a test, we want 2.
  const int coarse_dof = 2;

  // How many times to refine. 
  const int n_refine = 6; // (64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)

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
    cout << "Processing level " << i << "\n";
    // Update to the new lattice size.
    curr_x_len /= x_block;
    curr_y_len /= y_block;

    // Create a new lattice object.
    lats[i] = new Lattice2D(curr_x_len, curr_y_len, coarse_dof);

    // Create a new null vector. These are copied into local memory in the
    // transfer object, so we can create and destroy these in this loop.
    complex<double>* null_vector = allocate_vector<complex<double>>(lats[i-1]->get_size_cv());
    constant_vector(null_vector, 1.0, lats[i-1]->get_size_cv());
    complex<double>* null_vector2 = allocate_vector<complex<double>>(lats[i-1]->get_size_cv());
    constant_vector(null_vector2, 1.0, lats[i-1]->get_size_cv()/2);
    constant_vector(null_vector2+lats[i-1]->get_size_cv()/2, -1.0, lats[i-1]->get_size_cv()/2);
    complex<double>** null_vectors = new complex<double>*[2];
    null_vectors[0] = null_vector;
    null_vectors[1] = null_vector2;

    // Create and populate a transfer object.
    // Fine lattice, coarse lattice, null vector(s), perform the block ortho.
    transfer_objs[i-1] = new TransferMG(lats[i-1], lats[i], null_vectors, true);

    // Push a new level on the multigrid object! Also, save the global null vector.
    // Arg 1: New lattice
    // Arg 2: New transfer object (between new and prev lattice)
    // Arg 3: Should we construct the coarse stencil?
    // Arg 4: What should we construct the coarse stencil from? (Not relevant yet.)
    // Arg 5: Non-block-orthogonalized null vector.
    mg_object->push_level(lats[i], transfer_objs[i-1], true, MultigridMG::QMG_MULTIGRID_PRECOND_ORIGINAL, null_vectors);

    //mg_object->get_stencil(i)->print_stencil_site(0,0,"Site: ");

    // Clean up local vector, since they get copied in.
    deallocate_vector(&null_vectors[0]);
    deallocate_vector(&null_vectors[1]);
    delete[] null_vectors; 
  }

  // Build a MultigridStorage object. This just allocates static memory
  // for the recursive solve so we aren't allocating/deallocating space
  // as we recurse. Eventually the MultigridStorage object will go
  // into the MultigridMG class. 
  // Let's pretend we need 6 vectors, we'll fix that later. 
  MultigridStorage* mg_storage = new MultigridStorage(lats, mg_object->get_num_levels(), 6);

  // Compare each coarse stencil with prolong, apply, restrict.
  for (i = 1; i <= n_refine; i++)
  {
    // Get a vector to store a rhs in.
    complex<double>* rhs = mg_storage->get_temp_vector(i, 0);
    gaussian(rhs, lats[i]->get_size_cv(), generator);

    // Get a vector to store the lhs from applying the stencil in.
    complex<double>* apply_lhs = mg_storage->get_temp_vector(i, 1);
    zero_vector(apply_lhs, lats[i]->get_size_cv());
    mg_object->apply_stencil(apply_lhs, rhs, i);
    double norm = sqrt(norm2sq(apply_lhs, lats[i]->get_size_cv()));
    //cout << "Level " << i << " apply_lhs norm is " << norm << "\n";

    // Get a vector to store the prolonged rhs in.
    complex<double>* rhs_pro = mg_storage->get_temp_vector(i-1, 0);
    zero_vector(rhs_pro, lats[i-1]->get_size_cv());
    mg_object->prolong_c2f(rhs, rhs_pro, i-1);

    // Get a vector to apply the fine stencil to.
    complex<double>* Arhs_pro = mg_storage->get_temp_vector(i-1, 1);
    zero_vector(Arhs_pro, lats[i-1]->get_size_cv());
    mg_object->apply_stencil(Arhs_pro, rhs_pro, i-1);

    // Get a vector to restrict into.
    complex<double>* proAres_lhs = mg_storage->get_temp_vector(i, 2);
    zero_vector(proAres_lhs, lats[i]->get_size_cv());
    mg_object->restrict_f2c(Arhs_pro, proAres_lhs, i-1);
    //cout << "Level " << i << " proAres_lhs norm is " << sqrt(norm2sq(proAres_lhs, lats[i]->get_size_cv())) << "\n";

    // Compare.
    cout << "Level " << i << " build has comparison norm " << sqrt(diffnorm2sq(apply_lhs, proAres_lhs, lats[i]->get_size_cv()))/norm << "\n";
  }


  ///////////////
  // Clean up. //
  ///////////////

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