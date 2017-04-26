// Copyright (c) 2017 Evan S Weinberg
// Header file for various reductions, such as on timeslices.

#include <complex>
#include <random>

using std::complex; 
using std::polar;

#ifndef QMG_REDUCTIONS
#define QMG_REDUCTIONS

// QMG
#include "lattice/lattice.h"

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Perform a norm2sq on each timeslice of a color vector.
// This admits a more efficient implementation for Nd > 2.
// (Nd == 2 has issues because "y" is intertwined with "x".)
template<typename T> inline void norm2sq_cv_timeslice(T* sum, complex<T>* cv, Lattice2D* lat)
{
  int i;
  int x, y, c;
  const int nt = lat->get_dim_mu(lat->get_nd()-1);
  const int size_cv = lat->get_size_cv();
  complex<T> tmp;
  T norm2;

  for (i = 0; i < nt; i++)
    sum[i] = 0.0;

  for (i = 0; i < size_cv; i++)
  {
    tmp = cv[i];
    norm2 = std::real(std::conj(tmp)*tmp);
    lat->cv_index_to_coord(i, x, y, c);
    sum[y] = sum[y] + norm2;
  }
}

// Perform a re_dot on each timeslice of a color vector.
// This admits a more efficient implementation for Nd > 2.
// (Nd == 2 has issues because "y" is intertwined with "x".)
template<typename T> inline void redot_cv_timeslice(T* sum, complex<T>* cv1, complex<T>* cv2, Lattice2D* lat)
{
  int i;
  int x, y, c;
  const int nt = lat->get_dim_mu(lat->get_nd()-1);
  const int size_cv = lat->get_size_cv();
  complex<T> tmp1, tmp2;
  T dot_val;

  for (i = 0; i < nt; i++)
    sum[i] = 0.0;

  for (i = 0; i < size_cv; i++)
  {
    tmp1 = cv1[i];
    tmp2 = cv2[i];
    dot_val = std::real(std::conj(tmp1)*tmp2);
    lat->cv_index_to_coord(i, x, y, c);
    sum[y] = sum[y] + dot_val;
  }
}

// Perform a dot on each timeslice of a color vector.
// This admits a more efficient implementation for Nd > 2.
// (Nd == 2 has issues because "y" is intertwined with "x".)
template<typename T> inline void redot_cv_timeslice(complex<T>* sum, complex<T>* cv1, complex<T>* cv2, Lattice2D* lat)
{
  int i;
  int x, y, c;
  const int nt = lat->get_dim_mu(lat->get_nd()-1);
  const int size_cv = lat->get_size_cv();
  complex<T> tmp1, tmp2;
  complex<T> dot_val;

  for (i = 0; i < nt; i++)
    sum[i] = 0.0;

  for (i = 0; i < size_cv; i++)
  {
    tmp1 = cv1[i];
    tmp2 = cv2[i];
    dot_val = std::conj(tmp1)*tmp2;
    lat->cv_index_to_coord(i, x, y, c);
    sum[y] = sum[y] + dot_val;
  }
}

#endif // QMG_REDUCTIONS

