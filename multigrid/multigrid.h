// Copyright (c) 2017 Evan S Weinberg
// Header file for a multigrid object, which contains everything (?)
// needed for a multigrid preconditioner.

// NOT YET COMPLETED, WILL NOT COMPILE

// MG requires:
// * Knowledge of the number of levels.
// * Knowledge of each lattice (stored as a std::vector of Lattice2D*)
// * Knowledge of each transfer object (stored as a std::vector of TransferMG*)
// * Knowledge of the fine level stencil (stored as the first element of
//    a std::vector of Stencil2D*)
// ** Optionally knows lower levels, if they've been constructed. (If there,
//     it's stored in the above std::vector of Stencil2D*. If not there, stored
//     as a zero pointer in the vector.)
// * Is written such that one level is pushed at a time.
// ** This allows the user to start storing levels in the MG object when
//     recursively generating coarser levels.
// * A private function to (optionaly) explicitly build the coarse stencil.
// ** Implemented as a flag on pushing a new level.
// ** Should have a flag to specify if it should be built from the
//     original operator or right block Jacobi operator.
// * A function to apply the stencil at a specified level.
// * A function to prepare, solve, reconstruct right block Jacobi and, where
//    possible, Schur preconditioned systems. 
// * Optional, but for convenience, two pre-allocated temporary vectors
//    at each level.
// ** This avoids allocating and deallocating temporary vectors, such as for
//     the preconditioned functions.
// * Optional, but for convenience, the ability to store non-block-orthogonalized
//    null vectors (perhaps for updating or projecting null vectors)

#ifndef QMG_MULTIGRID_OBJECT
#define QMG_MULTIGRID_OBJECT

#include <complex>
#include <cmath>
#include <iostream>
#include <vector>

using std::complex;
using std::vector;

// QLINALG
#include "blas/generic_vector.h"

// QMG
#include "lattice/lattice.h"
#include "stencil/stencil_2d.h"
#include "transfer/transfer.h"

class MultigridMG
{
private:
  // Get rid of copy, assignment operator.
  MultigridMG(TransferMG const &);
  MultigridMG& operator=(TransferMG const &);

  // Current number of levels. Gets updated when user
  // adds another layer. There's a public function that
  // exposes this. 
  int num_levels;

  // Knowledge of each lattice. Should have length = num_levels.
  vector<Lattice2D*> lattice_list;

  // Knowledge of each transfer operator. Should have length = num_levels - 1.
  vector<TransferMG*> transfer_list;

  // Knowledge of each stencil. Required to contain fine level stencil,
  // optionally contains coarser stencils (otherwise the element will be
  // set as a zero pointer). Should have length = num_levels.
  vector<Stencil2D*> stencil_list;

  // Optional, but for convenience, two pre-allocated temporary vectors
  // at each level. Should have length = num_levels.
  vector<complex<double>*> temp_vec_1;
  vector<complex<double>*> temp_vec_2;

  // Optionally, but for convenience, the ability to store non-block-
  // orthogonalized null vectors. Should have length = num_levels - 1.
  vector<complex<double>**> global_null_vectors; 

  
public:
  
  // Constructor. Takes in fine and coarse lattice, input null vectors, and
  // a flag about performing block orthonormalization (default true).
  // If the null vectors are pre-block orthonormalized, it might make sense
  // to not waste time re-orthonormalizing them. 
  TransferMG(Lattice2D* in_fine_lat, Lattice2D* in_coarse_lat, complex<double>** in_null_vectors, bool do_block_ortho = true)
    : fine_lat(in_fine_lat), coarse_lat(in_coarse_lat), const_num_null_vec(coarse_lat->get_nc()),
      const_coarse_volume(coarse_lat->get_volume()), blocksizes(0), coarse_map(0), null_vectors(0)
  {
    // Assume the vector construction failed until we know it hasn't.
    is_init = false;

    // Learn the blocksizes. 
    blocksizes = new int[fine_lat->get_nd()];
    fine_sites_per_coarse = fine_lat->get_nc();
    for (int i = 0; i < fine_lat->get_nd(); i++)
    {
      if (fine_lat->get_dim_mu(i) % coarse_lat->get_dim_mu(i) != 0)
      {
        std::cout << "[QMG-ERROR]: Fine lattice dimension " << i << "isn't divided evenly by coarse dimension.\n";
        return;
      }
      blocksizes[i] = fine_lat->get_dim_mu(i)/coarse_lat->get_dim_mu(i);
      fine_sites_per_coarse *= blocksizes[i];
    }

    // Prepare the one-to-many mapping between coarse sites and fine sites+dof.
    coarse_map = allocate_vector<int*>(coarse_lat->get_volume());
    build_mapping();

    // Allocate null vectors, copy them in.
    null_vectors = new complex<double>*[coarse_lat->get_nc()];
    for (int i = 0; i < coarse_lat->get_nc(); i++)
    {
      null_vectors[i] = allocate_vector<complex<double>>(fine_lat->get_size_cv());
      copy_vector(null_vectors[i], in_null_vectors[i], fine_lat->get_size_cv());
    }

    // Block orthonormalize, if requested.
    if (do_block_ortho)
      block_orthonormalize();

    // We're good!
    is_init = true;

  }

  // Destructor. Clean up!
  ~TransferMG()
  {
    if (blocksizes != 0) { delete[] blocksizes; }
    if (null_vectors != 0)
    {
      for (int i = 0; i < const_num_null_vec; i++)
      {
        if (null_vectors[i] != 0) { deallocate_vector(&null_vectors[i]); }
      }
      delete[] null_vectors;
    }
    if (coarse_map != 0)
    {
      for (int i = 0; i < const_coarse_volume; i++)
      {
        if (coarse_map[i] != 0) { delete[] coarse_map[i]; }
      }
      deallocate_vector(&coarse_map);
    }
  }

  // Public function exposing number of levels.
  int get_num_levels()
  {
    return num_levels;
  }

  // Query if the transfer object is properly initialized.
  bool is_initialized()
  {
    return is_init;
  }

  // The public 'prolong' operator (coarse_to_fine)
  // Arg 1: coarse vector.
  // Arg 2: fine vector.
  void prolong_c2f(complex<double>* coarse_cv, complex<double>* fine_cv)
  {
    prolong_c2f(coarse_cv, fine_cv, null_vectors, coarse_lat->get_nc());
  }

  // The public 'restrict' operator (fine to coarse)
  // Arg 1: fine vector.
  // Arg 2: coarse vector.
  void restrict_f2c(complex<double>* fine_cv, complex<double>* coarse_cv)
  {
    restrict_f2c(fine_cv, coarse_cv, null_vectors, coarse_lat->get_nc());
  }


private:

  // Declare merge sort for build_mapping. Smallest int first.
  void merge_sort(int* sort_me, int* temp, int size);

  // Recursive function to learn every fine site corresponding
  // to a coarse site. Recursive b/c it's dimension-independent!
  void recursive_site_build(int* list, int* coarse_coord, int* coords, int step, int& count);

  // Special C function to take the inverse
  // square root of the real part of a site. 
  // Needed for block orthonormalize.
  static inline void inv_real_sqrt(int i, complex<double>& elem, void* extra_data)
  {
    double tmp = 1.0/sqrt(real(elem));
    elem = complex<double>(tmp, 0.0);
  }
  
};

// Recursive function to learn every fine site corresponding
// to a coarse site. Recursive b/c it's dimension-independent!
void TransferMG::recursive_site_build(int* list, int* coarse_coords, int* coords, int step, int& count)
{
  if (step < fine_lat->get_nd())
  {
    // Loop over every site in block in dimension step.
    for (int i = coarse_coords[step]*blocksizes[step]; i < (coarse_coords[step]+1)*blocksizes[step]; i++)
    {
      coords[step] = i;
      recursive_site_build(list, coarse_coords, coords, step+1, count);
    }
  }
  else // we're on the lowest level. coords is fully populated.
  {
    for (int i = 0; i < fine_lat->get_nc(); i++)
    {
      list[count++] = fine_lat->cv_coord_to_index(coords, i);
    }
  }
}

// Figure out the one-to-many mapping between coarse sites (volume)
//         and fine sites (full color vector).
//         There's no reason why this couldn't be many-to-many (for overlapping 
//         domains, for ex).
void TransferMG::build_mapping()
{
  // Get total number of coarse sites.
  const int coarse_volume = coarse_lat->get_volume();

  // Temporary memory used for merge sort.
  int* tmp_mem = new int[fine_sites_per_coarse];

  // Temporary memory for recursive fine site build.
  int* coarse_coord = new int[coarse_lat->get_nd()];
  int* fine_coords = new int[coarse_lat->get_nd()];

  // Counter.
  int count; 

  // Loop over coarse sites.
  // OPENMP/ACC WOULD GO HERE.
  for (int i = 0; i < coarse_volume; i++)
  {
    // Initialize coarse map.
    coarse_map[i] = new int[fine_sites_per_coarse];

    // Get coarse coordinates.
    coarse_lat->index_to_coord(i, coarse_coord);

    // Recursively build fine site list.
    count = 0;
    recursive_site_build(coarse_map[i], coarse_coord, fine_coords, 0, count);

    // Sort the fine indices.
    merge_sort(coarse_map[i], tmp_mem, fine_sites_per_coarse);
  }

  // Clean up.
  delete[] tmp_mem;
  delete[] coarse_coord;
  delete[] fine_coords; 

}

// More general 'prolong' operator (coarse_to_fine)
// Arg 1: coarse vector.
// Arg 2: fine vector.
// Arg 3: null vector(s).
// Arg 4: number of null vectors. (there are some cases we want to prolong w/ 1.)
void TransferMG::prolong_c2f(complex<double>* coarse_cv, complex<double>* fine_cv, complex<double>** in_null_vectors, int nvec)
{
  int i,i_dof,j;
  const int coarse_volume = coarse_lat->get_volume();

  // Loop over every coarse site.
  for (i = 0; i < coarse_volume; i++)
  {
    // Loop over each null vector (generically, coarse dof)
    for (i_dof = 0; i_dof < nvec; i_dof++)
    {
      // Get the coarse color vector index.
      int cv_index = coarse_lat->vol_index_dof_to_cv_index(i, i_dof);

      // We saved each coarse site's corresponding fine degrees of freedom.
      for (j = 0; j < fine_sites_per_coarse; j++)
      {
        // Do the prolong!
        // Update the fine vector site with the i_dof's null vector site times
        // the relevant coarse index. 
        fine_cv[coarse_map[i][j]] += in_null_vectors[i_dof][coarse_map[i][j]]*coarse_cv[cv_index];
      }

    }
  }
}

// Pre-declaring a 'restrict' operator (fine_to_coarse)
// Arg 1: fine vector.
// Arg 2: coarse vector.
// Arg 3: null vector(s).
// Arg 4: number of null vectors. (there are some cases we want to restrict w/ 1.)
void TransferMG::restrict_f2c(complex<double>* fine_cv, complex<double>* coarse_cv, complex<double>** in_null_vectors, int nvec)
{
  int i,i_dof,j;
  const int coarse_volume = coarse_lat->get_volume();

  // Loop over every coarse site.
  for (i = 0; i < coarse_volume; i++)
  {
    // Loop over each null vector (generically, coarse dof)
    for (i_dof = 0; i_dof < nvec; i_dof++)
    {
      // Get the coarse color vector index.
      int cv_index = coarse_lat->vol_index_dof_to_cv_index(i, i_dof);

      // We saved each coarse site's corresponding fine degrees of freedom.
      for (j = 0; j < fine_sites_per_coarse; j++)
      {
        // Do the restrict.
        // Update the coarse vector site with the conjugate i_dof's null
        // vector times the relevant fine index.
        coarse_cv[cv_index] += conj(in_null_vectors[i_dof][coarse_map[i][j]])*fine_cv[coarse_map[i][j]];
      }
    }
  }
}

// Perform the block orthonormalization.
void TransferMG::block_orthonormalize()
{
  // James Osborn has a smart way of doing it---you can write block
  // orthonormalization in terms of prolong-restrict operations. 
  // Ref: https://github.com/usqcd-software/qopqdp/blob/master/lib/mg/mg_p.c
  // Line 240, function QOPP(mgOrtho)(QDPN(ColorVector) *cv[], int nv, QOP_MgBlock *mgb)

  // Iterators.
  int i,j;

  // Values.
  const int fine_size_cv = fine_lat->get_size_cv();
  const int coarse_size_cv = coarse_lat->get_size_cv();
  const int coarse_dof = coarse_lat->get_nc();

  // Temporary vectors. 

  // Fine...
  complex<double>* fine_cv_1 = allocate_vector<complex<double>>(fine_size_cv);
  zero_vector(fine_cv_1, fine_size_cv);

  // ...and coarse.
  complex<double>* coarse_cv_2 = allocate_vector<complex<double>>(coarse_size_cv);
  zero_vector(coarse_cv_2, coarse_size_cv);

  // Loop over all null vectors.
  for (i = 0; i < coarse_dof; i++)
  {
    // Block Orthogonalize null vector i against previous null vectors.
    for (j = 0; j < i; j++)
    {
      // Zero out vectors.
      zero_vector(fine_cv_1, fine_size_cv);
      zero_vector(coarse_cv_2, coarse_size_cv);

      // Restrict null vector i with (already ortho'd null vector) j.
      // This is the <\vec{i},\vec{j}> dot product in Gram-Schmidt.
      restrict_f2c(null_vectors[i], coarse_cv_2, &null_vectors[j], 1);

      // Prolong again with null vector i.
      // This forms <\vec{i},\vec{j}> \vec{j}.
      prolong_c2f(coarse_cv_2, fine_cv_1, &null_vectors[j], 1);

      // Subtract off.
      // This forms \vec{i} -= <\vec{i},\vec{j}> \vec{j}.
      caxpy(-1.0, fine_cv_1, null_vectors[i], fine_size_cv);
    }

    // Block normalize null vector i.
    // zero out vectors again.
    zero_vector(fine_cv_1, fine_size_cv);
    zero_vector(coarse_cv_2, coarse_size_cv);

    // Restrict null vector i with null vector i.
    // This is a block <\vec{i}, \vec{i}>.
    restrict_f2c(null_vectors[i], coarse_cv_2, &null_vectors[i], 1);

    // Prepare to normalize. This requires taking the inverse sqrt of
    // <\vec{i}, \vec{i}>. The '0' is because it doesn't need any info passed in.
    arb_local_function_vector(coarse_cv_2, inv_real_sqrt, 0, coarse_size_cv);

    // Prolong the inverse norm with null vector i. 
    // This is block \vec{i}/sqrt(<\vec{i}, \vec{i}>).
    prolong_c2f(coarse_cv_2, fine_cv_1, &null_vectors[i], 1);

    // Set this back to null_vectors[i].
    copy_vector(null_vectors[i], fine_cv_1, fine_size_cv);
  }

  // Clean up.
  deallocate_vector(&coarse_cv_2);
  deallocate_vector(&fine_cv_1);
}

////////////////////////
// AUXILARY FUNCTIONS //
////////////////////////

// Declare merge sort for build_mapping. Smallest int first.
void TransferMG::merge_sort(int* sort_me, int* temp, int size)
{
  if (size == 1) // trivial
  {
    return;
  }
  if (size == 2) // trivial
  {
    temp[0] = sort_me[0];
    if (sort_me[1] < sort_me[0])
    {
      sort_me[0] = sort_me[1];
      sort_me[1] = temp[0];
    }
    return;
  }

  // recurse.
  merge_sort(sort_me, temp, size/2);
  merge_sort(sort_me + size/2, temp + size/2, size - size/2);

  // merge
  int curr1 = 0;
  int curr2 = size/2;
  int currtmp = 0;

  while (curr1 < size/2 && curr2 < size)
  {
    if (sort_me[curr1] < sort_me[curr2])
      temp[currtmp++] = sort_me[curr1++];
    else
      temp[currtmp++] = sort_me[curr2++];
  }

  while (curr1 < size/2)
    temp[currtmp++] = sort_me[curr1++];

  while (curr2 < size)
    temp[currtmp++] = sort_me[curr2++];

  for (int i = 0; i < size; i++)
    sort_me[i] = temp[i];
}

#endif // QMG_MULTIGRID_OBJECT
