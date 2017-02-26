// Copyright (c) 2017 Evan S Weinberg
// A bit of an exploration place for coding up a good
// prolong/restrict interface.

#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>

using namespace std;

// QLINALG
#include "blas/generic_vector.h"

// QMG
#include "lattice/lattice.h"
#include "cshift/cshift_2d.h"
#include "u1/u1_utils.h"
#include "stencil/stencil_2d.h"

#include "operators/gaugedlaplace.h"

// Merge sort wooo, smallest element first.
void merge_sort(int* sort_me, int* temp, int size)
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

int main(int argc, char** argv)
{
  // Set output to not be that long.
  cout << setiosflags(ios::fixed) << setprecision(6);

  //Iterators and such.
  int i,j;
  int x_coarse,y_coarse,x_fine,y_fine;
  int count; 

  // Basic information for fine level.
  const int x_len = 64;
  const int y_len = 64;
  const int dof = 1;

  // Blocking size.
  const int x_block = 4;
  const int y_block = 4;

  // Number of null vectors (equiv. degrees of freedom on coarse level)
  const int coarse_dof = 2;

  // Create a lattice object for the fine lattice.
  Lattice2D* lat = new Lattice2D(x_len, y_len, dof);

  // Create a lattice object for the coarse lattice.
  Lattice2D* coarse_lat = new Lattice2D(x_len/x_block, y_len/y_block, coarse_dof);

  // Step 1: Figure out the one-to-many mapping between coarse sites and fine sites.
  //         There's no reason why this couldn't be many-to-many (for overlapping 
  //         domains, for ex).

  // Get total number of coarse sites.
  const int coarse_volume = coarse_lat->get_volume();

  // Allocate a coarse volume array of pointers to lists of fine sites...
  int** coarse_map = allocate_vector<int*>(coarse_volume);

  // Compute number of fine sites per coarse site.
  const int fine_sites_per_coarse = x_block*y_block;

  // temporary memory needed for merge sorts.
  int* tmp_mem = new int[fine_sites_per_coarse];

  // I need some type of blas-like function for this...
  for (i = 0; i < coarse_volume; i++)
  {
    // We need to allocate each element in coarse_map as an array of length
    // 'fine_sites_per_coarse'.
    coarse_map[i] = new int[fine_sites_per_coarse];

    // Actually do the loop and find out which fine sites correspond to which
    // coarse sites.

    // Get coarse coordinates.
    coarse_lat->index_to_coord(i, x_coarse, y_coarse);

    // Loop over corresponding fine coordinates.
    // Thought: there must be some recursive way to do this for general
    // dimensions. 
    count = 0;
    for (y_fine = y_coarse*y_block; y_fine < (y_coarse+1)*y_block; y_fine++)
    {
      for (x_fine = x_coarse*x_block; x_fine < (x_coarse+1)*x_block; x_fine++)
      {
        coarse_map[i][count++] = lat->coord_to_index(x_fine, y_fine);
      }
    }

    // Sort the fine indices for efficiency. 
    merge_sort(coarse_map[i], tmp_mem, fine_sites_per_coarse);

  }
  // Clean up temporary memory.
  delete[] tmp_mem;

  // Good! Now we have the mapping between the coarse site at memory index
  // 'i' and the corresponding memory indices for the fine sites. We need this
  // to do a prolong and restrict.

  // Step 2: Create null vectors. For now, we'll just have two: one for
  //         even sites only, the other for all sites (intentionally 
  //         non-orthogonal).

  // Step 3: Prolong. We'll put something on a coarse site, and prolong it
  //         to the fine lattice.

  // Step 4: Restrict. We'll restrict the result from the fine lattice back
  //         down to the coarse lattice. This should match the original
  //         result, up to a normalization!

  // Step 5: Block orthonormalization. We need to block orthonormalize the
  //         vectors. As a test, we'll prolong and restrict again, and this
  //         time we should match the original coarse value.

  // Last step: Clean up. 
  for (i = 0; i < coarse_volume; i++)
  {
    delete[] coarse_map[i];
  }
  deallocate_vector(&coarse_map);
  delete lat;
  delete coarse_lat;

  return 0;
}