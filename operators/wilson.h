// Copyright (c) 2017 Evan S Weinberg
// Test of a gauged wilson op implementation.

#ifndef QMG_WILSON
#define QMG_WILSON

#include <iostream>
#include <complex>
using std::complex;

#include "blas/generic_vector.h"
#include "lattice/lattice.h"
#include "stencil/stencil_2d.h"


// Set up a gauged Wilson operator.
// This largely just requires overloading the 
// constructor, plus supplying a few convenience functions. 
struct Wilson2D : public Stencil2D
{
protected:
  // Get rid of copy, assignment.
  Wilson2D(Wilson2D const &);
  Wilson2D& operator=(Wilson2D const &);

  // Temporary space for eo solve.
  complex<double>* tmp_eo_space;

  double wilson_coeff;

public:

  // Predeclare function to assign links.
  void update_links(complex<double>* gauge_links);

  // Base constructor.
  Wilson2D(Lattice2D* in_lat, complex<double> mass, complex<double>* gauge_links, double wilson_coeff = 1.0)
    : Stencil2D(in_lat, QMG_PIECE_CLOVER_HOPPING, mass, 0.0, 0.0), wilson_coeff(wilson_coeff)
  {
    if (lat->get_nc() != 2)
    {
      std::cout << "[QMG-ERROR]: Wilson2D only supports Nc = 2.\n";
      return;
    }

    // Allocate temporary space.
    tmp_eo_space = allocate_vector<complex<double>>(lat->get_size_cv());

    // Prepare links.
    update_links(gauge_links);
  }

  ~Wilson2D()
  {
    deallocate_vector(&tmp_eo_space);
  }

};

// update gauge links.
void Wilson2D::update_links(complex<double>* gauge_links)
{
  // Prepare for complex numbers.
  const complex<double> cplxI(0.0,1.0);

  // Get the color matrix size. We stride with this when we
  // use the blas'.
  const int nc2 = lat->get_nc()*lat->get_nc();

  // Get the volume (number of sites).
  const int volume = lat->get_volume();


  // The clover term is the 2*wilson_term on the identity.
  constant_vector_blas(clover, nc2, 2.0*wilson_coeff, volume);
  constant_vector_blas(clover+1, nc2, 0.0, volume);
  constant_vector_blas(clover+2, nc2, 0.0, volume);
  constant_vector_blas(clover+3, nc2, 2.0*wilson_coeff, volume);


  // The hopping term is a bit more complicated, but not by much!
  const int cm_size = lat->get_size_cm();

  // Get the step for the nc = 1 gauge fields... oi.
  const int u1_cm_size = cm_size/nc2; 

  // +x
  // structure: 0.5*{{-w, 1},{1,-w}}*U_x(x)
  caxy_blas(-0.5*wilson_coeff, gauge_links, 1, hopping, nc2, volume);
  caxy_blas(0.5, gauge_links, 1, hopping+1, nc2, volume);
  caxy_blas(0.5, gauge_links, 1, hopping+2, nc2, volume);
  caxy_blas(-0.5*wilson_coeff, gauge_links, 1, hopping+3, nc2, volume);

  // +y
  // structure: 0.5*{{-w, -I},{I,-w}}
  caxy_blas(-0.5*wilson_coeff, gauge_links+u1_cm_size, 1, hopping+cm_size, nc2, volume);
  caxy_blas(-0.5*cplxI, gauge_links+u1_cm_size, 1, hopping+cm_size+1, nc2, volume);
  caxy_blas(0.5*cplxI, gauge_links+u1_cm_size, 1, hopping+cm_size+2, nc2, volume);
  caxy_blas(-0.5*wilson_coeff, gauge_links+u1_cm_size, 1, hopping+cm_size+3, nc2, volume);

  // -x requires a cshift, conj.
  // structure: 0.5*{{-w, -1},{-1, -w}}
  cshift(priv_cmatrix, gauge_links, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVENODD, 1, lat);
  conj_vector(priv_cmatrix, u1_cm_size);
  caxy_blas(-0.5*wilson_coeff, priv_cmatrix, 1, hopping+2*cm_size, nc2, volume);
  caxy_blas(-0.5, priv_cmatrix, 1, hopping+2*cm_size+1, nc2, volume);
  caxy_blas(-0.5, priv_cmatrix, 1, hopping+2*cm_size+2, nc2, volume);
  caxy_blas(-0.5*wilson_coeff, priv_cmatrix, 1, hopping+2*cm_size+3, nc2, volume);

  // -y requires a cshift, conj
  // structure: 0.5*{{-w, I},{-I,-w}}
  cshift(priv_cmatrix, gauge_links + u1_cm_size, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVENODD, 1, lat);
  conj_vector(priv_cmatrix, u1_cm_size);
  caxy_blas(-0.5*wilson_coeff, priv_cmatrix, 1, hopping+3*cm_size, nc2, volume);
  caxy_blas(0.5*cplxI, priv_cmatrix, 1, hopping+3*cm_size+1, nc2, volume);
  caxy_blas(-0.5*cplxI, priv_cmatrix, 1, hopping+3*cm_size+2, nc2, volume);
  caxy_blas(-0.5*wilson_coeff, priv_cmatrix, 1, hopping+3*cm_size+3, nc2, volume);
}


#endif