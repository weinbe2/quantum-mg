// Copyright (c) 2017 Evan S Weinberg
// Header file for generic 2d stencils, the rock of the code. 

#ifndef QMG_STENCIL_2D
#define QMG_STENCIL_2D

#include <iostream>
#include <complex>
#include "../lattice/lattice.h"
#include "blas/generic_vector.h"
#include "blas/generic_local_matrix.h"
#include "blas/generic_matrix.h"
#include "../cshift/cshift_2d.h"

// Indexing offset of, for ex, XP1, is
// lat->get_size_cm() * QMG_DIR_INDEX_XP1.
// Helpful for accessing certain sub-directions
// as a LatticeColorMatrix.
enum stencil_dir_index
{
  QMG_DIR_INDEX_0 = 0,
  QMG_DIR_INDEX_XP1 = 0,
  QMG_DIR_INDEX_YP1 = 1,
  QMG_DIR_INDEX_XM1 = 2,
  QMG_DIR_INDEX_YM1 = 3,
  QMG_DIR_INDEX_XP2 = 0,
  QMG_DIR_INDEX_YP2 = 1,
  QMG_DIR_INDEX_XM2 = 2,
  QMG_DIR_INDEX_YM2 = 3,
  QMG_DIR_INDEX_XP1YP1 = 0,
  QMG_DIR_INDEX_XM1YP1 = 1,
  QMG_DIR_INDEX_XM1YM1 = 2,
  QMG_DIR_INDEX_XP1YM1 = 3,
};

// Enum for pieces of stencil that may exist. 
enum stencil_pieces
{
  QMG_PIECE_CLOVER = 1,
  QMG_PIECE_HOPPING = 2,
  QMG_PIECE_TWOLINK = 4,
  QMG_PIECE_CORNER = 8,
  QMG_PIECE_CLOVER_HOPPING = 3,
  QMG_PIECE_TWOLINK_CORNER = 12,
  QMG_PIECE_ALL = 15,
};

// Enum for if a stencil has chirality or not.
enum chirality_state
{
  QMG_CHIRAL_NO = 0,
  QMG_CHIRAL_YES = 1,
  QMG_CHIRAL_UNKNOWN = 2 // used for coarse operator.
};

struct Stencil2D
{
protected:
  // Get rid of copy, assignment operator.
  Stencil2D(Stencil2D const &);
  Stencil2D& operator=(Stencil2D const &);

  // Internal memory for cshifts.
  complex<double>* priv_cmatrix;
  complex<double>* priv_cvector;

public:
  // Associated lattice!
  Lattice2D* lat; 
  
  // The following pieces are either allocated or set equal to 0. 
  
  // Nc x Nc x X x Y
  complex<double>* clover;
  
  // Nc x Nc x X x Y x {+X,+Y,-X,-Y} 
  complex<double>* hopping;
  
  // X x Y x Nc x Nc x {+2X, +2Y, -2X, -2Y}
  complex<double>* twolink;
  
  // X x Y x Nc x Nc x {+X+Y, -X+Y, -X-Y, +X-Y}
  complex<double>* corner;  
  
  // Have we generated a lattice?
  bool generated;
  
  // Identity shift, think local mass term. 
  complex<double> shift; 
  
  // Even/odd shift (minus sign on odd), think mass term in g5_staggered
  complex<double> eo_shift; 
  
  // Top/bottom half shift (minus sign on bottom half of color dof, think mass term on coarse g5_staggered.
  // i.e., top half of dof gets a lhs[i] += dof_shift*rhs[i], bottom half gets a rhs[i] -= dof_shift*rhs[i]
  complex<double> dof_shift;
    
  // Base constructor
  Stencil2D(Lattice2D* in_lat, int pieces, complex<double> in_shift = 0.0, complex<double> in_eo_shift = 0.0, complex<double> in_dof_shift = 0.0)
    : lat(in_lat), shift(in_shift), eo_shift(in_eo_shift), dof_shift(in_dof_shift)
  {
    generated = false;
    
    if (pieces & QMG_PIECE_CLOVER)
    {
      clover = allocate_vector<complex<double>>(lat->get_size_cm());
    }
    else
    {
      clover = 0;
    }
      
    if (pieces & QMG_PIECE_HOPPING)
    {
      hopping = allocate_vector<complex<double>>(lat->get_size_hopping());
    }
    else
    {
      hopping = 0;
    }
    
    if (pieces & QMG_PIECE_TWOLINK)
    {
      twolink = allocate_vector<complex<double>>(lat->get_size_hopping());
    }
    else
    {
      twolink = 0;
    }
      
    if (pieces & QMG_PIECE_CORNER)
    {
      corner = allocate_vector<complex<double>>(lat->get_size_corner());
    }
    else
    {
      corner = 0;
    }

    // Allocate private memory.
    priv_cmatrix = allocate_vector<complex<double>>(lat->get_size_cm());
    priv_cvector = allocate_vector<complex<double>>(lat->get_size_cv());
    
  }
  
  virtual ~Stencil2D()
  {
    if (clover != 0) { deallocate_vector(&clover); }
    if (hopping != 0) { deallocate_vector(&hopping); }
    if (twolink != 0) { deallocate_vector(&twolink); }
    if (corner != 0) { deallocate_vector(&corner); }

    // Deallocate private memory.
    deallocate_vector(&priv_cmatrix);
    deallocate_vector(&priv_cvector);
    
    generated = false; 

  }
  
  // Clear out stencils!
  void clear_stencils()
  {
    if (clover != 0) { zero_vector(clover, lat->get_size_cm()); }
    if (hopping != 0) { zero_vector(hopping, lat->get_size_hopping()); }
    if (twolink != 0) { zero_vector(twolink, lat->get_size_hopping()); }
    if (corner != 0) { zero_vector(corner, lat->get_size_corner()); }
    
    generated = false; 
  }
  
  // Prune pieces of the stencil. This deletes pieces.
  void prune_stencils(int pieces)
  {
    if (pieces & QMG_PIECE_CLOVER)
    {
      if (clover != 0) { deallocate_vector(&clover); }
    }
      
    if (pieces & QMG_PIECE_HOPPING)
    {
      if (hopping != 0) { deallocate_vector(&hopping); }
    }
    
    if (pieces & QMG_PIECE_TWOLINK)
    {
      if (twolink != 0) { deallocate_vector(&twolink); }
    }
      
    if (pieces & QMG_PIECE_CORNER)
    {
      if (corner != 0) { deallocate_vector(&corner); }
    }
    
    if (clover == 0 && hopping == 0 && twolink == 0 && corner == 0)
      generated = false;
      
  }
  
  // Prune pieces of the stencil if the max norm is below some tolerance.
  void try_prune_stencils(int pieces, double tol)
  {
    if (pieces & QMG_PIECE_CLOVER)
    {
      if (clover != 0 && norminf(clover, lat->get_size_cm()) < tol) { deallocate_vector(&clover); }
    }
      
    if (pieces & QMG_PIECE_HOPPING)
    {
      if (hopping != 0 && norminf(hopping, lat->get_size_hopping()) < tol) { deallocate_vector(&hopping); }
    }
    
    if (pieces & QMG_PIECE_TWOLINK)
    {
      if (twolink != 0 && norminf(twolink, lat->get_size_hopping()) < tol) { deallocate_vector(&twolink); }
    }
      
    if (pieces & QMG_PIECE_CORNER)
    {
      if (corner != 0 && norminf(corner, lat->get_size_corner()) < tol) { deallocate_vector(&corner); }
    }
    
    if (clover == 0 && hopping == 0 && twolink == 0 && corner == 0)
      generated = false;
  }
  
  // Print the full stencil at one site.
  void print_stencil_site(int x, int y, string prefix = "")
  {
    int i,j;
    int index; 
    const int nc = lat->get_nc();
      
    if (shift != 0.0)
    {
      cout << prefix << "Shift " << shift << "\n";
    }
      
    if (eo_shift != 0.0)
    {
      cout << prefix << "EO-Shift " << eo_shift << "\n";
    }
      
    if (dof_shift != 0.0)
    {
      cout << prefix << "DOF-Shift " << dof_shift << "\n";
    }
    
    if (clover != 0)
    {
      cout << prefix << "Clover\n";
      index = lat->cm_coord_to_index(x, y, 0, 0);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << clover[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
    }
    
    if (hopping != 0)
    {
      cout << prefix << "Hopping +x\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XP1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << hopping[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Hopping +y\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_YP1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << hopping[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Hopping -x\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XM1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << hopping[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Hopping -y\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_YM1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << hopping[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
    }
      
    if (twolink != 0)
    {
      cout << prefix << "Twolink +2x\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XP2);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << twolink[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Hopping +2y\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_YP2);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << twolink[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Hopping -2x\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XM2);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << twolink[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Hopping -2y\n";
      index = lat->hopping_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_YM2);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << twolink[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
    }
      
    if (corner != 0)
    {
      cout << prefix << "Corner +x+y\n";
      index = lat->corner_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XP1YP1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << corner[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Corner -x+y\n";
      index = lat->corner_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XM1YP1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << corner[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Corner -x-y\n";
      index = lat->corner_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XM1YM1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << corner[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
        
      cout << prefix << "Corner +x-y\n";
      index = lat->corner_coord_to_index(x, y, 0, 0, QMG_DIR_INDEX_XP1YM1);
      for (i = 0; i < nc; i++)
      {
        cout << prefix;
        for (j = 0; j < nc; j++)
        {
          cout << corner[index+i*nc+j] << " ";
        }
        cout << "\n";
      }
    }
  }
      
  // Need functions to apply M_{clover}, M_{eo}, M_{oe}, M_{twolink}, M_{corner}, M_{shift}
  //   and, of course, all. 

  void apply_M_clover(complex<double>* lhs, complex<double>* rhs)
  {
    if (clover == 0)
      return;

    const int nc = lat->get_nc();
    const int vol = lat->get_volume();

    cMATxpy(clover, rhs, lhs, vol, nc, nc);
  }

  // Apply the eo part of the stencil.
  void apply_M_eo(complex<double>* lhs, complex<double>* rhs)
  {
    if (hopping == 0)
    {
      cout << "[QMG-WARNING]: Attempt to call 'apply_M_eo' without hopping term.\n";
      return;
    }

    const int nc = lat->get_nc();
    const int half_vol = lat->get_volume()/2;
    const int size_cm = lat->get_size_cm();

    // + xhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_ODD, nc, lat);
    cMATxpy(hopping, priv_cvector, lhs, half_vol, nc, nc);

    // + yhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_ODD, nc, lat);
    cMATxpy(hopping + size_cm, priv_cvector, lhs, half_vol, nc, nc);

    // - xhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_ODD, nc, lat);
    cMATxpy(hopping + 2*size_cm, priv_cvector, lhs, half_vol, nc, nc);

    // - yhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_ODD, nc, lat);
    cMATxpy(hopping + 3*size_cm, priv_cvector, lhs, half_vol, nc, nc);
  }

  // Apply the eo part of the stencil for one direction.
  void apply_M_eo(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (hopping == 0)
    {
      cout << "[QMG-WARNING]: Attempt to call 'apply_M_eo' without hopping term.\n";
      return;
    }

    const int nc = lat->get_nc();
    const int half_vol = lat->get_volume()/2;
    const int size_cm = lat->get_size_cm();

    switch (dir)
    {
      case QMG_DIR_INDEX_XP1: // + xhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_ODD, nc, lat);
        cMATxpy(hopping, priv_cvector, lhs, half_vol, nc, nc);
        break; 

      case QMG_DIR_INDEX_YP1: // + yhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_ODD, nc, lat);
        cMATxpy(hopping + size_cm, priv_cvector, lhs, half_vol, nc, nc);
        break;

      case QMG_DIR_INDEX_XM1: // - xhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_ODD, nc, lat);
        cMATxpy(hopping + 2*size_cm, priv_cvector, lhs, half_vol, nc, nc);
        break;

      case QMG_DIR_INDEX_YM1: // - yhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_ODD, nc, lat);
        cMATxpy(hopping + 3*size_cm, priv_cvector, lhs, half_vol, nc, nc);
        break;
    }
  }
  
  // Apply the oe part of the stencil.
  void apply_M_oe(complex<double>* lhs, complex<double>* rhs)
  {
    if (hopping == 0)
    {
      cout << "[QMG-WARNING]: Attempt to call 'apply_M_oe' without hopping term.\n";
      return;
    }

    const int nc = lat->get_nc();
    const int half_vol = lat->get_volume()/2;
    const int half_cv = lat->get_size_cv()/2;
    const int size_cm = lat->get_size_cm();
    const int half_cm = lat->get_size_cm()/2;

    // + xhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVEN, nc, lat);
    cMATxpy(hopping + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);

    // + yhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVEN, nc, lat);
    cMATxpy(hopping + size_cm + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);

    // - xhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVEN, nc, lat);
    cMATxpy(hopping + 2*size_cm + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);

    // - yhat
    cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVEN, nc, lat);
    cMATxpy(hopping + 3*size_cm + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);
  }

  // Apply the oe part of the stencil for one direction.
  void apply_M_oe(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (hopping == 0)
    {
      cout << "[QMG-WARNING]: Attempt to call 'apply_M_oe' without hopping term.\n";
      return;
    }

    const int nc = lat->get_nc();
    const int half_vol = lat->get_volume()/2;
    const int half_cv = lat->get_size_cv()/2;
    const int size_cm = lat->get_size_cm();
    const int half_cm = lat->get_size_cm()/2;

    switch (dir)
    {
      case QMG_DIR_INDEX_XP1: // + xhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XP1, QMG_EO_FROM_EVEN, nc, lat);
        cMATxpy(hopping + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);
        break; 

      case QMG_DIR_INDEX_YP1: // + yhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YP1, QMG_EO_FROM_EVEN, nc, lat);
        cMATxpy(hopping + size_cm + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);
        break;

      case QMG_DIR_INDEX_XM1: // - xhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_XM1, QMG_EO_FROM_EVEN, nc, lat);
        cMATxpy(hopping + 2*size_cm + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);
        break;

      case QMG_DIR_INDEX_YM1: // - yhat
        cshift(priv_cvector, rhs, QMG_CSHIFT_FROM_YM1, QMG_EO_FROM_EVEN, nc, lat);
        cMATxpy(hopping + 3*size_cm + half_cm, priv_cvector + half_cv, lhs + half_cv, half_vol, nc, nc);
        break;
    }
  }

  void apply_M_hopping(complex<double>* lhs, complex<double>* rhs)
  {
    if (hopping != 0)
    {
      apply_M_eo(lhs, rhs);
      apply_M_oe(lhs, rhs);
    }
  }

  void apply_M_hopping(complex<double>* lhs, complex<double>* rhs, stencil_dir_index dir)
  {
    if (hopping != 0)
    {
      apply_M_eo(lhs, rhs, dir);
      apply_M_oe(lhs, rhs, dir);
    }
  }
  
  // void apply_M_twolink(complex<double>* lhs, complex<double>* rhs);
  // void apply_M_corner(complex<double>* lhs, complex<double>* rhs);
  
  // Apply the shifts (think a mass term, maybe with signs a la \gamma_5 D)
  void apply_M_shift(complex<double>* lhs, complex<double>* rhs)
  {
    // Get size of a LatticeColorVector
    int cv = lat->get_size_cv();

    if (lat->get_volume() == 1) // This corner case is annoying.
    {
      if (lat->get_nc() % 2 == 0)
      {
        for (int c = 0; c < lat->get_nc()/2; c++)
        {
          lhs[c] += (shift+eo_shift+dof_shift)*rhs[c];
          lhs[c+lat->get_nc()/2] += (shift+eo_shift-dof_shift)*rhs[c];
        }
      }
      else
      {
        for (int c = 0; c < lat->get_nc(); c++)
        {
          lhs[c] += (shift+eo_shift)*rhs[c];
        }
      }
      return;
    }

    // Apply shift and eo_shift to even sites.
    caxpy(shift+eo_shift, rhs, lhs, cv/2);

    // Apply shift and eo_shift to odd sites.
    caxpy(shift-eo_shift, rhs+cv/2, lhs+cv/2, cv/2);

    // There's no good way to do the dof shift with the current mem layout...
    if (dof_shift != 0.0 && lat->get_nc() % 2 == 0) // is dof_shift valid?
    {
      int nc = lat->get_nc();
      for (int c = 0; c < nc/2; c++)
      {
        // Shift top half of degrees of freedom by shift_dof.
        caxpy_stride(dof_shift, rhs, lhs, lat->get_size_cv(), c, nc);

        // Shift bottom half of degrees of freedom by -shift_dof.
        caxpy_stride(-dof_shift, rhs, lhs, lat->get_size_cv(), c+nc/2, nc);
      }
    }
  }

  void apply_M(complex<double>* lhs, complex<double>* rhs)
  {
    if (clover != 0)
    {
      apply_M_clover(lhs, rhs);
    }

    if (hopping != 0)
    {
      apply_M_eo(lhs, rhs);
      apply_M_oe(lhs, rhs);
    }

    if (twolink != 0)
    {
      cout << "[QMG-WARNING]: two link stencil not yet supported.\n";
    }

    if (corner != 0)
    {
      cout << "[QMG-WARNING]: corner stencil not yet supported.\n";
    }

    apply_M_shift(lhs, rhs);
  }
    
  // Need functions to build dagger of a stencil from another stencil,
  //   and build a normal stencil from two one-link stencils.
  // void build_M_dagger_stencil(Stencil2D* orig_stenc); 
  // void build_M_dagger_M_stencil(Stencil2D* orig_stenc);


  // Functions to return the shifts.
  complex<double> get_shift()
  {
    return shift;
  }

  complex<double> get_shift_eo()
  {
    return eo_shift;
  }

  complex<double> get_shift_dof()
  {
    return dof_shift;
  }

public:
  // Abstract static functions.

  // Return the number of degrees of freedom.
  // 'i' is for the case of Ls where the number
  // of degrees of freedom may depend on some value
  // passed in.
  static int get_dof(int i = 0)
  {
    return -1;
  }

  // Return true if the operator has a sense of "chirality",
  // false otherwise.
  static chirality_state has_chirality()
  {
    return QMG_CHIRAL_UNKNOWN;
  }

  // A few ways to perform chiral projections.

  // In place project onto up (true), down (false)
  virtual void chiral_projection(complex<double>* vector, bool is_up) = 0;

  // Copy projection onto up, down.
  virtual void chiral_projection_copy(complex<double>* orig, complex<double>* dest, bool is_up) = 0;

  // Copy the down projection into a new vector, perform the up in place.
  virtual void chiral_projection_both(complex<double>* orig_to_up, complex<double>* down) = 0;

};

// Special C function wrappers for stencil applications.
void apply_stencil_2D_M(complex<double>* lhs, complex<double>* rhs, void* extra_data)
{
  Stencil2D* stenc = (Stencil2D*)extra_data;
  zero_vector(lhs, stenc->lat->get_size_cv());
  stenc->apply_M(lhs, rhs); // lhs = M rhs
}

#endif // QMG_STENCIL_2D