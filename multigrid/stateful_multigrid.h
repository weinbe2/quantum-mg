// Copyright (c) 2017 Evan S Weinberg
// Header file for a stateful multigrid object,
// which keeps the state of an actual multigrid solve.
// It's also aware of how to perform smearing, preconditioning,
// etc on each level.

// To do: Add a counter for the number of op applications at each level.
// This way we can track the number of operator applications over 
// the entire solve. 

// QLINALG
#include "blas/generic_vector.h"
#include "inverters/generic_gcr.h"
#include "inverters/generic_gcr_var_precond.h"

// QMG
#include "stencil/stencil_2d.h"
#include "multigrid/multigrid.h"


// A class that inherits from MultigridMG, adding extra functions
// which track the state of an MG solve, as well as what type of 
// preconditioning to use at each level. 
class StatefulMultigridMG : public MultigridMG
{
private:
  // Get rid of copy, assignment.
  StatefulMultigridMG(StatefulMultigridMG const &);
  StatefulMultigridMG& operator=(StatefulMultigridMG const &);

  int current_level;

public: 

  // Structure that contains information about the solve
  // at a given level: how to perform the outer solve,
  // as well as the pre-smooth and post-smooth. This
  // object exists for level except the last, though the
  // outer solve info is ignored if a given level
  // is used as the outermost level in a recursive solve.
  // There is a separate object for the solve of the
  // coarsest level.
  struct LevelSolveMG
  {
    /* FINE OP */

    // Whether we're using the original, rbjacobi, or schur solve.
    QMGStencilType fine_stencil_app;

    /* OUTER SOLVE */

    // Tolerance for outer solve.
    double intermediate_tol;

    // Maximum number of iterations for outer solve.
    int intermediate_iters;

    // Restart frequency. -1 means don't restart.
    int intermediate_restart_freq;

    /* PRESMOOTHER */

    // Tolerance for presmoother (for a flexible presmoother)
    double pre_tol;

    // Number of iterations for presmoother
    int pre_iters;

    // What solver to use, relaxation params, other params,
    // preconditioning...

    /* POSTSMOOTHER */

    // Tolerance for postsmoother (for a flexible postsmoother)
    double post_tol;

    // Number of iterations for presmoother
    int post_iters;

    // By default, there's "no" stopping condition.
    LevelSolveMG()
      : fine_stencil_app(QMG_MATVEC_ORIGINAL), 
        intermediate_tol(1e-20), intermediate_iters(10000000),
        intermediate_restart_freq(32),
        pre_tol(1e-20), pre_iters(1000000),
        post_tol(1e-20), post_iters(1000000)
    { ; }

  };

  // Structure that describes how to do the coarsest solve.
  struct CoarsestSolveMG
  {
    /* FINE OP */

    // Whether we're using the original, rbjacobi, schur,
    // CGNE, or CGNR solve.
    QMGStencilType coarsest_stencil_app;

    /* OUTER SOLVE */

    // Tolerance for outer solve.
    double coarsest_tol;

    // Maximum number of iterations for outer solve.
    int coarsest_iters;

    // Restart frequency. -1 means don't restart.
    int coarsest_restart_freq;

    // By default, there's "no" stopping condition.
    CoarsestSolveMG()
      : coarsest_stencil_app(QMG_MATVEC_ORIGINAL),
        coarsest_tol(1e-20), coarsest_iters(100000000),
        coarsest_restart_freq(32)
    { ; }
  };

protected:

  // Vector of LevelSolveMG. Should be of length mg_object->get_num_levels()-1.
  vector<LevelSolveMG*> level_solve_list;

  // Information on the coarsest level solve.
  CoarsestSolveMG* coarsest_solve; 

public:

  // Simple constructor.
  StatefulMultigridMG(Lattice2D* in_lat, Stencil2D* in_stencil, CoarsestSolveMG* in_coarsest_solve)
    : MultigridMG(in_lat, in_stencil), coarsest_solve(in_coarsest_solve)
  {

    // Set the current level to zero.
    current_level = 0;
  }

  // Set the multigrid level.
  void set_multigrid_level(int level)
  {
    if (level >= 0 && level < get_num_levels())
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
    if (current_level < get_num_levels()-2)
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

  // Get the LevelSolveMG structure for a given fine level.
  LevelSolveMG* get_level_solve(int i)
  {
    if (i >= 0 && i < get_num_levels()-1 && level_solve_list[i] != 0)
    {
      return level_solve_list[i];
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: LevelSolveMG level " << i << " does not exist in StatefulMultigridMG object.\n";
      return 0;
    }
  }

  LevelSolveMG* get_level_solve()
  {
    return get_level_solve(current_level);
  }

  // Get the CoarsestSolveMG structure.
  CoarsestSolveMG* get_coarsest_solve()
  {
    return coarsest_solve;
  }

  // Virtual overloads of "push_level" which set the level_solve to a zero pointer...
  void push_level(Lattice2D* new_lat, TransferMG* new_transfer, bool build_stencil = false, bool is_chiral = false, QMGMultigridPrecondStencil build_stencil_from = QMG_MULTIGRID_PRECOND_ORIGINAL, CoarseOperator2D::QMGCoarseBuildStencil build_extra = CoarseOperator2D::QMG_COARSE_BUILD_ORIGINAL, complex<double>** nvecs = 0)
  {
    MultigridMG::push_level(new_lat, new_transfer, build_stencil, is_chiral, build_stencil_from, build_extra, nvecs);
    level_solve_list.push_back(0); // oiiii
  }

  void push_level(Lattice2D* new_lat, TransferMG* new_transfer, bool build_stencil = false, bool is_chiral = false, QMGMultigridPrecondStencil build_stencil_from = QMG_MULTIGRID_PRECOND_ORIGINAL, complex<double>** nvecs = 0)
  {
    MultigridMG::push_level(new_lat, new_transfer, build_stencil, is_chiral, build_stencil_from, nvecs);
    level_solve_list.push_back(0); // oiiii
  }

  void push_level(Lattice2D* new_lat, TransferMG* new_transfer,complex<double>** nvecs)
  {
    MultigridMG::push_level(new_lat, new_transfer, nvecs);
    level_solve_list.push_back(0); // oiiii
  }

  // Safe versions of push_level which also push a new LevelSolveMG object.
  // See multigrid.h for the purpose of the other arguments! 
  void push_level(Lattice2D* new_lat, TransferMG* new_transfer, LevelSolveMG* in_solve, bool build_stencil = false, bool is_chiral = false, QMGMultigridPrecondStencil build_stencil_from = QMG_MULTIGRID_PRECOND_ORIGINAL, CoarseOperator2D::QMGCoarseBuildStencil build_extra = CoarseOperator2D::QMG_COARSE_BUILD_ORIGINAL, complex<double>** nvecs = 0)
  {
    MultigridMG::push_level(new_lat, new_transfer, build_stencil, is_chiral, build_stencil_from, build_extra, nvecs);

    // Push the level info. An outer solve can only
    // be QMG_MATVEC_ORIGINAL, QMG_MATVEC_RIGHT_JACOBI, QMG_MATVEC_RIGHT_SCHUR.
    if (in_solve->fine_stencil_app != QMG_MATVEC_ORIGINAL &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_JACOBI &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_SCHUR)
    {
      std::cout << "[QMG-ERROR]: In StatefulMultigridMG:;push_level, LevelSolveMG::fine_stencil_app should only be original, right jacobi, or schur.\n";
    }

    level_solve_list.push_back(in_solve);
  }

  void push_level(Lattice2D* new_lat, TransferMG* new_transfer, LevelSolveMG* in_solve, bool build_stencil = false, bool is_chiral = false, QMGMultigridPrecondStencil build_stencil_from = QMG_MULTIGRID_PRECOND_ORIGINAL, complex<double>** nvecs = 0)
  {
    MultigridMG::push_level(new_lat, new_transfer, build_stencil, is_chiral, build_stencil_from, nvecs);

    // Push the level info. An outer solve can only
    // be QMG_MATVEC_ORIGINAL, QMG_MATVEC_RIGHT_JACOBI, QMG_MATVEC_RIGHT_SCHUR.
    if (in_solve->fine_stencil_app != QMG_MATVEC_ORIGINAL &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_JACOBI &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_JACOBI)
    {
      std::cout << "[QMG-ERROR]: In StatefulMultigridMG:;push_level, LevelSolveMG::fine_stencil_app should only be original, right jacobi, or schur.\n";
    }

    level_solve_list.push_back(in_solve);
  }

  void push_level(Lattice2D* new_lat, TransferMG* new_transfer, LevelSolveMG* in_solve, complex<double>** nvecs)
  {
    MultigridMG::push_level(new_lat, new_transfer, nvecs);

    // Push the level info. An outer solve can only
    // be QMG_MATVEC_ORIGINAL, QMG_MATVEC_RIGHT_JACOBI, QMG_MATVEC_RIGHT_SCHUR.
    if (in_solve->fine_stencil_app != QMG_MATVEC_ORIGINAL &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_JACOBI &&
          in_solve->fine_stencil_app != QMG_MATVEC_RIGHT_JACOBI)
    {
      std::cout << "[QMG-ERROR]: In StatefulMultigridMG:;push_level, LevelSolveMG::fine_stencil_app should only be original, right jacobi, or schur.\n";
    }

    level_solve_list.push_back(in_solve);
  }

  // Overloaded version of pop_level which also pops the level_solve_list.
  void pop_level()
  {
    MultigridMG::pop_level();

    // Pop level solve info.
    level_solve_list.pop_back();
  }

  static void mg_preconditioner(complex<double>* lhs, complex<double>* rhs, int size, void* extra_data, inversion_verbose_struct* verb)
  {
    // Expose the Multigrid objects.

    // State.
    StatefulMultigridMG* mg_object = (StatefulMultigridMG*)extra_data;
    int level = mg_object->get_multigrid_level();
    //cout << "Entered level " << level << "\n" << flush;

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

    // Verbosity and inversion info. Currently this is hacky.
    inversion_info invif;
    inversion_verbose_struct verb2(VERB_SUMMARY, std::string(" "));
    if (verb->verbosity == VERB_NONE)
    {
      verb2.verbosity = VERB_NONE;
      verb2.precond_verbosity = VERB_NONE;
    }
    else
    {
      verb2.precond_verbosity = VERB_SUMMARY;
    }
    verb2.verb_prefix = "  ";
    for (int i = 1; i < level+1; i++)
      verb2.verb_prefix += "  ";
    verb2.verb_prefix += "[QMG-MG-SOLVE-INFO]: Level " + to_string(level+1) + " ";

    // Get info on pre- and post-smooth.
    LevelSolveMG* level_solve = mg_object->get_level_solve();
    if (level_solve == 0)
    {
      std::cout << "[QMG-MG-SOLVE-ERROR]: Level solve for level " << level << " does not exist.\n";
      return;
    }
    int n_pre_smooth = level_solve->pre_iters;
    double pre_smooth_tol = level_solve->pre_tol;
    int n_post_smooth = level_solve->post_iters;
    double post_smooth_tol = level_solve->post_tol;

    // Function for what type of fine apply we need to do.
    QMGStencilType fine_stencil_type = level_solve->fine_stencil_app;
    matrix_op_cplx apply_fine_M = Stencil2D::get_apply_function(level_solve->fine_stencil_app);

    // The fine solve depends on if we're doing a schur solve
    // or not.
    int fine_size_solve = fine_size;
    if (fine_stencil_type == QMG_MATVEC_RIGHT_SCHUR)
      fine_size_solve /= 2;

    // Learn about coarse solve.
    int coarse_max_iter;
    double coarse_tol;
    int coarse_restart;
    QMGStencilType coarse_stencil_type;
    if (level < total_num_levels-2)
    {
      coarse_stencil_type = mg_object->get_level_solve(level+1)->fine_stencil_app;
      coarse_max_iter = mg_object->get_level_solve(level+1)->intermediate_iters;
      coarse_tol = mg_object->get_level_solve(level+1)->intermediate_tol;
      coarse_restart = mg_object->get_level_solve(level+1)->intermediate_restart_freq;
    }
    else // coarsest solve.
    {
      coarse_stencil_type = mg_object->get_coarsest_solve()->coarsest_stencil_app;
      coarse_max_iter = mg_object->get_coarsest_solve()->coarsest_iters;
      coarse_tol = mg_object->get_coarsest_solve()->coarsest_tol;
      coarse_restart = mg_object->get_coarsest_solve()->coarsest_restart_freq;
    }
    // Function for what type of coarse apply we need to do.
    matrix_op_cplx apply_coarse_M = Stencil2D::get_apply_function(coarse_stencil_type);

    // The coarse size depends on if we're doing a schur solve
    // or not.
    int coarse_size_solve = coarse_size;
    if (coarse_stencil_type == QMG_MATVEC_RIGHT_SCHUR)
      coarse_size_solve /= 2;

    // We need a temporary vector for mat-vecs everywhere.
    complex<double>* Atmp = fine_storage->check_out();

    // Step 1: presmooth.
    // Solve A z1 = rhs, form new residual r1 = rhs - A z1
    complex<double>* z1 = fine_storage->check_out();
    zero_vector(z1, fine_size);
    minv_vector_gcr_restart(z1, rhs, fine_size_solve, n_pre_smooth, pre_smooth_tol, coarse_restart, apply_fine_M, (void*)fine_stencil);
    zero_vector(Atmp, fine_size);
    fine_stencil->apply_M(Atmp, z1, fine_stencil_type);
    complex<double>* r1 = fine_storage->check_out();
    caxpbyz(1.0, rhs, -1.0, Atmp, r1, fine_size_solve);

    // Next stop: restrict, prep for coarse solve, recurse (or coarsest solve), prolong.
    complex<double>* r_coarse = coarse_storage->check_out();
    zero_vector(r_coarse, coarse_size);
    transfer->restrict_f2c(r1, r_coarse);
    fine_storage->check_in(r1);
    complex<double>* r_coarse_prep = coarse_storage->check_out();
    zero_vector(r_coarse_prep, coarse_size);
    coarse_stencil->prepare_M(r_coarse_prep, r_coarse, coarse_stencil_type);
    complex<double>* e_coarse = coarse_storage->check_out();
    zero_vector(e_coarse, coarse_size);
    if (level == total_num_levels-2) // if we're already on the coarsest level
    {
      if (coarse_restart == -1)
      {
        invif = minv_vector_gcr(e_coarse, r_coarse_prep, coarse_size_solve,
                          coarse_max_iter, coarse_tol, 
                          apply_coarse_M, (void*)coarse_stencil, &verb2);
      }
      else
      {
        invif = minv_vector_gcr_restart(e_coarse, r_coarse_prep, coarse_size_solve,
                          coarse_max_iter, coarse_tol, coarse_restart, 
                          apply_coarse_M, (void*)coarse_stencil, &verb2);
      }
    }
    else
    {
      // Recurse.
      mg_object->go_coarser();
      // K cycle
      if (coarse_restart == -1)
      {
        invif = minv_vector_gcr_var_precond(e_coarse, r_coarse_prep, coarse_size_solve,
                          coarse_max_iter, coarse_tol,
                          apply_coarse_M, (void*)coarse_stencil,
                          mg_preconditioner, (void*)mg_object, &verb2);
      }
      else
      {
        invif = minv_vector_gcr_var_precond_restart(e_coarse, r_coarse_prep, coarse_size_solve,
                          coarse_max_iter, coarse_tol, coarse_restart,
                          apply_coarse_M, (void*)coarse_stencil,
                          mg_preconditioner, (void*)mg_object, &verb2);
      }
      // V cycle
      //mg_preconditioner(e_coarse, r_coarse, coarse_size, (void*)stateful_mg_object);
      mg_object->go_finer();
    }
    //cout << "Level " << level << " coarse preconditioner took " << invif.iter << " iterations.\n" << flush;
    coarse_storage->check_in(r_coarse_prep);
    complex<double>* e_coarse_reconstruct = coarse_storage->check_out();
    zero_vector(e_coarse_reconstruct, coarse_size);
    coarse_stencil->reconstruct_M(e_coarse_reconstruct, e_coarse, r_coarse, coarse_stencil_type);

    /*complex<double>* tmp = coarse_storage->check_out();
    zero_vector(tmp, coarse_size);
    coarse_stencil->apply_M(tmp, e_coarse_reconstruct);
    cout << "Coarse solve level " << level << " has norm2sqdiff " << sqrt(diffnorm2sq(tmp, r_coarse, coarse_size))/sqrt(norm2sq(r_coarse, coarse_size)) << "\n";
    coarse_storage->check_in(tmp);*/
    
    coarse_storage->check_in(r_coarse);
    coarse_storage->check_in(e_coarse);
    complex<double>* z2 = fine_storage->check_out();
    zero_vector(z2, fine_size);
    transfer->prolong_c2f(e_coarse_reconstruct, z2);
    if (coarse_stencil_type == QMG_MATVEC_RIGHT_SCHUR) { zero_vector(z2 + fine_size/2, fine_size/2); }
    coarse_storage->check_in(e_coarse_reconstruct);
    zero_vector(lhs, fine_size_solve);
    cxpyz(z1, z2, lhs, fine_size_solve);
    fine_storage->check_in(z1);
    fine_storage->check_in(z2);

    // Last stop, post smooth. Form r2 = r - A(z1 + z2) = r - Ae, solve A z3 = r2.
    zero_vector(Atmp, fine_size);
    fine_stencil->apply_M(Atmp, lhs, fine_stencil_type);
    complex<double>* r2 = fine_storage->check_out();
    caxpbyz(1.0, rhs, -1.0, Atmp, r2, fine_size_solve);
    complex<double>* z3 = fine_storage->check_out();
    zero_vector(z3, fine_size);
    minv_vector_gcr(z3, r2, fine_size_solve, n_post_smooth, post_smooth_tol, apply_fine_M, (void*)fine_stencil);
    cxpy(z3, lhs, fine_size_solve);

    // Check vectors back in.
    fine_storage->check_in(Atmp);
    fine_storage->check_in(r2);
    fine_storage->check_in(z3);

    //cout << "Exited level " << level << "\n" << flush;
  }

};
