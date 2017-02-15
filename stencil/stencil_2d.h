// Copyright (c) 2017 Evan S Weinberg
// Header file for generic 2d stencils, the rock of the code. 

#ifndef QMG_STENCIL_2D
#define QMG_STENCIL_2D

#include <iostream>
#include <complex>
#include "../lattice/lattice.h"
#include "../blas/generic_vector.h"

// Enum for all possible stencil directions.
// Largely used for generating additional refinements.
enum stencil_dir
{
  QMG_DIR_ALL = 0, // default, full stencil.
  QMG_DIR_0 = 1,   // clover
  QMG_DIR_XP1 = 2, // +x
  QMG_DIR_YP1 = 3, // +y
  QMG_DIR_XM1 = 4, // -x
  QMG_DIR_YM1 = 5, // -y
  QMG_DIR_XP2 = 6, // +2x
  QMG_DIR_XP1YP1 = 10, // +x+y
  QMG_DIR_YP2 = 7, // +2y
  QMG_DIR_XM1YP1 = 11, // -x+y
  QMG_DIR_XM2 = 8, // -2x
  QMG_DIR_XM1YM1 = 12, // -x-y
  QMG_DIR_YM2 = 9, // -2y
  QMG_DIR_XP1YM1 = 13, // +x-y
};

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
}

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
}

struct Stencil2D
{
private:
  // Get rid of copy, assignment operator.
  Stencil2D(Stencil2D const &);
  Stencil2D& operator=(Stencil2D const &);

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
  Stencil2D(Lattice2D* in_lat, int pieces, complex<double> in_shift = 0.0, complex<double> in_eo_shift = 0.0, complex<double> in_dof_shift)
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
    
  }
  
  ~stencil_2d()
  {
    if (clover != 0) { deallocate_vector(&clover); }
    if (hopping != 0) { deallocate_vector(&hopping); }
    if (twolink != 0) { deallocate_vector(&twolink); }
    if (corner != 0) { deallocate_vector(&corner); }
    
    generated = false; 

  }
  
  // Clear out stencils!
  void clear_stencils()
  {
    if (clover != 0) { zero<double>(clover, lat->get_size_cm()); }
    if (hopping != 0) { zero<double>(hopping, lat->get_size_hopping()); }
    if (twolink != 0) { zero<double>(twolink, lat->get_size_hopping()); }
    if (corner != 0) { zero<double>(corner, lat->get_size_corner()); }
    
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
    const int lattice_size = lat->get_lattice_size();
      
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
      
    // Need functions to apply M_{clover}, M_{eo}, M_{oe}, M_{twolink}, M_{corner}, M_{shift}
    //   and, of course, all. 
    // void apply_M_clover(complex<double>* lhs, complex<double>* rhs);
    // void apply_M_eo(complex<double>* lhs, complex<double>* rhs);
    // void apply_M_oe(complex<double>* lhs, complex<double>* rhs);
    // void apply_M_twolink(complex<double>* lhs, complex<double>* rhs);
    // void apply_M_corner(complex<double>* lhs, complex<double>* rhs);
    // void apply_M_shift(complex<double>* lhs, complex<double>* rhs);
    // void apply_M(complex<double>* lhs, complex<double>* rhs); 
      
    // Need functions to build dagger of a stencil from another stencil,
    //   and build a normal stencil from two one-link stencils.
    // void build_M_dagger_stencil(Stencil2D* orig_stenc); 
    // void build_M_dagger_M_stencil(Stencil2D* orig_stenc);
};


#endif // QMG_STENCIL_2D