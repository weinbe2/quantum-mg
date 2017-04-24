// Copyright (c) 2017 Evan S Weinberg
// Header file for a stateful multigrid object,
// which keeps the state of an actual multigrid solve.
// It's also aware of how to perform smearing, preconditioning,
// etc on each level.

// QLINALG
#include "blas/generic_vector.h"
#include "inverters/generic_gcr.h"
#include "inverters/generic_gcr_var_precond.h"

// QMG
#include "stencil/stencil_2d.h"
#include "multigrid/multigrid.h"


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

  // Structure that contains information about pre-smooth,
  // post-smooth, and coarse solve for each level.
  // Could eventually contain info on what preconditioner
  // to use, solving with CGNE, etc.
  // Could also support K cycle, W cycle...
  // This structure needs to exist for each fine level.
  struct LevelInfoMG
  {
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

    // What solver to use, relaxation params, other params,
    // preconditioning..

    /* COARSE SOLVE */

    // Tolerance for coarse solve (for a flexible coarse solver)
    double coarse_tol;

    // Number of iterations for a coarse solve.
    int coarse_iters;

    // Restart freq for a coarse solve. -1 means don't restart. 
    int coarse_restart_freq;

    // By default, there's "no" stopping condition.
    LevelInfoMG()
      : pre_tol(1e-20), pre_iters(1000000),
        post_tol(1e-20), post_iters(1000000),
        coarse_tol(1e-20), coarse_iters(1000000),
        coarse_restart_freq(32)
    { ; }

  };

private:

  // Vector of LevelInfoMG. Should be of length mg_object->get_num_levels()-1.
  vector<LevelInfoMG*> level_info_list;

public:

  // Simple constructor.
  StatefulMultigridMG(MultigridMG* mg_object, int current_level = 0)
    : mg_object(mg_object), current_level(current_level)
  {
    // Fill level info structure with nulls.
    // These need to be set to run an MG solve. 
    for (int i = 0; i < mg_object->get_num_levels()-1; i++)
    {
      level_info_list.push_back(0);
    }
  }

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
    if (current_level < mg_object->get_num_levels()-2)
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

  // Set the LevelInfoMG structure for a given fine level.
  void set_level_info(int i, LevelInfoMG* level_info)
  {
    if (i >= 0 && i < mg_object->get_num_levels()-1)
    {
      level_info_list[i] = level_info;
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: LevelInfo level " << i << " does not exist in StatefulMultigridMG object.\n";
    }
  }

  // Get the LevelInfoMG structure for a given fine level.
  LevelInfoMG* get_level_info(int i)
  {
    if (i >= 0 && i < mg_object->get_num_levels()-1 && level_info_list[i] != 0)
    {
      return level_info_list[i];
    }
    else
    {
      cout << "[QMG-ERROR]: Out of range: LevelInfo level " << i << " does not exist in StatefulMultigridMG object.\n";
      return 0;
    }
  }

  LevelInfoMG* get_level_info()
  {
    return get_level_info(current_level);
  }

  static void mg_preconditioner(complex<double>* lhs, complex<double>* rhs, int size, void* extra_data, inversion_verbose_struct* verb)
  {
    // Expose the Multigrid objects.

    // State.
    StatefulMultigridMG* stateful_mg_object = (StatefulMultigridMG*)extra_data;
    int level = stateful_mg_object->get_multigrid_level();
    cout << "Entered level " << level << "\n" << flush;

    // MultigridMG.
    MultigridMG* mg_object = stateful_mg_object->get_multigrid_object();

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

    // Verbosity and inversion info.
    inversion_info invif;
    inversion_verbose_struct verb2;
    verb2.verbosity = VERB_DETAIL;
    verb2.verb_prefix = "Level " + to_string(level+1) + ": ";
    verb2.precond_verbosity = VERB_DETAIL;
    verb2.precond_verb_prefix = "Prec ";

    // Get smoothing, coarse solve info structure for current level.
    LevelInfoMG* level_info = stateful_mg_object->get_level_info();

    int n_pre_smooth = level_info->pre_iters;
    double pre_smooth_tol = level_info->pre_tol;
    int n_post_smooth = level_info->post_iters;
    double post_smooth_tol = level_info->post_tol;
    int coarse_max_iter = level_info->coarse_iters;
    double coarse_tol = level_info->coarse_tol;
    int coarse_restart = level_info->coarse_restart_freq;

    // We need a temporary vector for mat-vecs everywhere.
    complex<double>* Atmp = fine_storage->check_out();

    // Step 1: presmooth.
    // Solve A z1 = rhs, form new residual r1 = rhs - A z1
    complex<double>* z1 = fine_storage->check_out();
    zero_vector(z1, fine_size);
    minv_vector_gcr_restart(z1, rhs, fine_size, n_pre_smooth, pre_smooth_tol, coarse_restart, apply_stencil_2D_M, (void*)fine_stencil);
    zero_vector(Atmp, fine_size);
    fine_stencil->apply_M(Atmp, z1);
    complex<double>* r1 = fine_storage->check_out();
    caxpbyz(1.0, rhs, -1.0, Atmp, r1, fine_size);

    // Next stop: restrict, recurse (or coarsest solve), prolong.
    complex<double>* r_coarse = coarse_storage->check_out();
    zero_vector(r_coarse, coarse_size);
    transfer->restrict_f2c(r1, r_coarse);
    fine_storage->check_in(r1);
    complex<double>* e_coarse = coarse_storage->check_out();
    zero_vector(e_coarse, coarse_size);
    if (level == total_num_levels-2) // if we're already on the coarsest level
    {
      // Do coarsest solve.
      verb2.verbosity = VERB_NONE;
      if (coarse_restart == -1)
      {
        invif = minv_vector_gcr(e_coarse, r_coarse, coarse_size,
                          coarse_max_iter, coarse_tol, 
                          apply_stencil_2D_M, (void*)coarse_stencil, &verb2);
      }
      else
      {
        invif = minv_vector_gcr_restart(e_coarse, r_coarse, coarse_size,
                          coarse_max_iter, coarse_tol, coarse_restart, 
                          apply_stencil_2D_M, (void*)coarse_stencil, &verb2);
      }
    }
    else
    {
      // Recurse.
      stateful_mg_object->go_coarser();
      // K cycle
      if (coarse_restart == -1)
      {
        invif = minv_vector_gcr_var_precond(e_coarse, r_coarse, coarse_size,
                          coarse_max_iter, coarse_tol,
                          apply_stencil_2D_M, (void*)coarse_stencil,
                          mg_preconditioner, (void*)stateful_mg_object, &verb2);
      }
      else
      {
        invif = minv_vector_gcr_var_precond_restart(e_coarse, r_coarse, coarse_size,
                          coarse_max_iter, coarse_tol, coarse_restart,
                          apply_stencil_2D_M, (void*)coarse_stencil,
                          mg_preconditioner, (void*)stateful_mg_object, &verb2);
      }
      // V cycle
      //mg_preconditioner(e_coarse, r_coarse, coarse_size, (void*)stateful_mg_object);
      stateful_mg_object->go_finer();
    }
    cout << "Level " << level << " coarse preconditioner took " << invif.iter << " iterations.\n" << flush;
    coarse_storage->check_in(r_coarse);
    complex<double>* z2 = fine_storage->check_out();
    zero_vector(z2, fine_size);
    transfer->prolong_c2f(e_coarse, z2);
    coarse_storage->check_in(e_coarse);
    zero_vector(lhs, fine_size);
    cxpyz(z1, z2, lhs, fine_size);
    fine_storage->check_in(z1);
    fine_storage->check_in(z2);

    // Last stop, post smooth. Form r2 = r - A(z1 + z2) = r - Ae, solve A z3 = r2.
    zero_vector(Atmp, fine_size);
    fine_stencil->apply_M(Atmp, lhs);
    complex<double>* r2 = fine_storage->check_out();
    caxpbyz(1.0, rhs, -1.0, Atmp, r2, fine_size);
    complex<double>* z3 = fine_storage->check_out();
    zero_vector(z3, fine_size);
    minv_vector_gcr(z3, r2, fine_size, n_post_smooth, post_smooth_tol, apply_stencil_2D_M, (void*)fine_stencil);
    cxpy(z3, lhs, fine_size);

    // Check vectors back in.
    fine_storage->check_in(Atmp);
    fine_storage->check_in(r2);
    fine_storage->check_in(z3);

    cout << "Exited level " << level << "\n" << flush;
  }

};
