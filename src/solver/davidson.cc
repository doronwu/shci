#include "davidson.h"
#include <algorithm>
#include <cmath>
#include <eigen/Eigen/Dense>
#include "../config.h"
#include <random>


/* ------------------------------------------------------------
   Helper: double Gram–Schmidt and norm check
   Adds vec to basis (v, Hv) if it survives orthonormalisation.
   Returns true  if the vector was accepted, false otherwise.
   ------------------------------------------------------------ */
static bool push_new_vector(std::vector<double>& vec,
                            const SparseMatrix&  H,
                            std::vector<std::vector<double>>& v,
                            std::vector<std::vector<double>>& Hv,
                            double lindep)
{
    const size_t dim = vec.size();
    /* two GS passes ------------------------------------------------ */
    for (int pass = 0; pass < 2; ++pass)
        for (size_t i = 0; i < v.size(); ++i) {
            double s = Util::dot_omp(vec, v[i]);
            #pragma omp parallel for
            for (size_t j = 0; j < dim; ++j) vec[j] -= s * v[i][j];
        }

    double nrm = std::sqrt(Util::dot_omp(vec, vec));
    if (nrm < lindep) return false;               // too small, reject

    #pragma omp parallel for
    for (size_t j = 0; j < dim; ++j) vec[j] /= nrm;

    /* store -------------------------------------------------------- */
    v.push_back(vec);
    Hv.push_back(H.mul(vec));
    return true;
}

/* ===================================================================
   Davidson::diagonalize  (block version with items 1‑3)
   =================================================================== */
void Davidson::diagonalize(const SparseMatrix&                       H,
                           const std::vector<std::vector<double>>&   initial_vectors,
                           double  tol_e,           /* 1e‑12 or tighter */
                           bool    verbose)
{
    const double tol_r = std::sqrt(tol_e);         // residual tolerance
    const double lindep = 1e-12;                   // drop threshold
    const std::size_t max_cycle = 50;

    /* -------- basic dimensions ----------------------------------- */
    const std::size_t dim       = initial_vectors[0].size();
    const std::size_t n_states  = std::min(dim, initial_vectors.size());

    /* -------- storage for Krylov basis and Ritz data ------------- */
    std::vector<std::vector<double>> v;            // basis vectors
    std::vector<std::vector<double>> Hv;           // H|v_i⟩
    v.reserve(64);  Hv.reserve(64);

    std::vector<std::vector<double>> w(n_states, std::vector<double>(dim));
    std::vector<std::vector<double>> Hw(n_states, std::vector<double>(dim));

    std::vector<double> evals(n_states, 0.0), evals_prev(n_states, 1e100);
    std::vector<double> resid_norm(n_states, 1e100);
    std::vector<char>   root_converged(n_states, 0);

    /* -------- pre‑compute diagonal for Jacobi preconditioner ----- */
    std::vector<double> diag(dim);
    for (std::size_t i = 0; i < dim; ++i) diag[i] = H.get_diag(i);

    /* -------- 0. start with orthonormalised user guesses --------- */
    for (std::size_t k = 0; k < n_states; ++k) {
        std::vector<double> g = initial_vectors[k];                 // copy
        push_new_vector(g, H, v, Hv, lindep);
    }

    /* -------- main Davidson loop --------------------------------- */
    std::size_t iter = 0;
    while (iter++ < max_cycle) {

        const std::size_t m = v.size();                             // basis size
        /* 1. build projected Hamiltonian -------------------------- */
        Eigen::MatrixXd Hm = Eigen::MatrixXd::Zero(m, m);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j <= i; ++j) {                  // lower tri
                double hij = Util::dot_omp(v[i], Hv[j]);
                Hm(i,j) = Hm(j,i) = hij;
            }

        /* 2. diagonalise projected H ------------------------------ */
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hm);
        const Eigen::VectorXd& lam = es.eigenvalues();
        const Eigen::MatrixXd& U   = es.eigenvectors();

        /* 3. back‑transform first n_states Ritz pairs ------------- */
        #pragma omp parallel for
        for (std::size_t k = 0; k < n_states; ++k) {
            evals[k] = lam(k);
            std::fill(w[k].begin(),  w[k].end(),  0.0);
            std::fill(Hw[k].begin(), Hw[k].end(), 0.0);

            for (std::size_t i = 0; i < m; ++i) {
                double c = U(i, k);
                const auto& vi  = v[i];
                const auto& Hvi = Hv[i];
                for (std::size_t j = 0; j < dim; ++j) {
                    w[k][j]  += c * vi [j];
                    Hw[k][j] += c * Hvi[j];
                }
            }
        }

        /* 4. residuals, convergence test, and new directions ------ */
        bool all_conv = true;
        for (std::size_t k = 0; k < n_states; ++k) {
            if (root_converged[k]) continue;

            /* residual r = H|w〉 – e|w〉 --------------------------- */
            std::vector<double> r(dim);
            #pragma omp parallel for
            for (std::size_t j = 0; j < dim; ++j)
                r[j] = Hw[k][j] - evals[k] * w[k][j];

            resid_norm[k] = std::sqrt(Util::dot_omp(r, r));

            bool conv_now =
                  (std::fabs(evals[k] - evals_prev[k]) < tol_e) &&
                  (resid_norm[k] < tol_r);

            root_converged[k] = conv_now;
            all_conv &= conv_now;

            if (!conv_now) {
                /* 5. Jacobi‑Davidson pre‑conditioning ------------- */
                #pragma omp parallel for
                for (std::size_t j = 0; j < dim; ++j) {
                    double denom = evals[k] - diag[j];
                    if (std::fabs(denom) < 1e-4)
                        denom = (denom >= 0 ? 1 : -1) * 1e-4;        // level‑shift
                    r[j] /= denom;
                }
                push_new_vector(r, H, v, Hv, lindep);               // may be rejected
            }
        }

        if (verbose) {
            std::printf("Dav %2zu :", iter);
            for (double e : evals) std::printf(" % .10f", e);
            std::printf("\n");
        }

        if (all_conv) break;
        evals_prev = evals;                                         // for Δe test:
    }

    /* -------- copy results out ---------------------------------- */
    lowest_eigenvalues  = evals;
    lowest_eigenvectors = w;
    converged           = std::all_of(root_converged.begin(),
                                      root_converged.end(),
                                      [](char c){ return c; });
}


void Davidson::diagonalize2(
    const SparseMatrix& matrix,
    const std::vector<std::vector<double>>& initial_vectors,
    const double target_error,
    const bool verbose) {
  const double TOLERANCE = target_error;
  const size_t N_ITERATIONS_STORE = 5; // storage per state.

  const size_t dim = initial_vectors[0].size();
  const unsigned n_states = std::min(dim, initial_vectors.size());
  for (auto& eigenvec : lowest_eigenvectors) eigenvec.resize(dim);

  if (dim == 1) {
    lowest_eigenvalues[0] = matrix.get_diag(0);
    lowest_eigenvectors[0].resize(1);
    lowest_eigenvectors[0][0] = 1.0;
    converged = true;
    return;
  }

  const size_t n_iterations_store = std::min(dim, N_ITERATIONS_STORE);
  std::vector<double> lowest_eigenvalues_prev(n_states, 0.0);

  std::vector<std::vector<double>> v(n_states * n_iterations_store);
  std::vector<std::vector<double>> Hv(n_states * n_iterations_store);
  std::vector<std::vector<double>> w(n_states);
  std::vector<std::vector<double>> Hw(n_states);
  for (size_t i = 0; i < v.size(); i++) {
    v[i].resize(dim);
  }

  for (size_t i = 0; i < n_states; i++) {
    w[i].resize(dim);
    Hw[i].resize(dim);
  }

  for (unsigned i_state = 0; i_state < n_states; i_state++) {
    double norm = sqrt(Util::dot_omp(initial_vectors[i_state], initial_vectors[i_state]));
#pragma omp parallel for
    for (size_t j = 0; j < dim; j++) v[i_state][j] = initial_vectors[i_state][j] / norm;
    if (i_state > 0) {
      // Orthogonalize
      double inner_prod;
      for (unsigned k_state = 0; k_state < i_state; k_state++) {
        inner_prod = Util::dot_omp(v[i_state], v[k_state]);
        for (size_t j = 0; j < dim; j++) v[i_state][j] -= inner_prod * v[k_state][j];
      }
      // Normalize
      norm = sqrt(Util::dot_omp(v[i_state], v[i_state]));
      for (size_t j = 0; j < dim; j++) v[i_state][j] /= norm;
    }
  }

  Eigen::MatrixXd h_krylov =
      Eigen::MatrixXd::Zero(n_states * n_iterations_store, n_states * n_iterations_store);
  Eigen::MatrixXd eigenvector_krylov =
      Eigen::MatrixXd::Zero(n_states * n_iterations_store, n_states * n_iterations_store);
  converged = false;
  size_t n_converged = 0;

  for (unsigned i_state = 0; i_state < n_states; i_state++) {
    Hv[i_state] = matrix.mul(v[i_state]);
    lowest_eigenvalues[i_state] = Util::dot_omp(v[i_state], Hv[i_state]);
    h_krylov(i_state, i_state) = lowest_eigenvalues[i_state];
    w[i_state] = v[i_state];
    Hw[i_state] = Hv[i_state];
  }
  if (verbose) {
    printf("Davidson #0:");
    for (const auto& eigenval : lowest_eigenvalues) printf("  %.10f", eigenval);
    printf("\n");
  }
  lowest_eigenvalues_prev = lowest_eigenvalues;

  size_t it_real = 1;
  size_t i_state_precond = 0; // state preconditioning on
  for (size_t it = n_states; it < n_iterations_store * n_states * 2; it++) {
    size_t it_circ = it % (n_states * n_iterations_store);
    if (it >= n_states * n_iterations_store) {
      if (it_circ < n_states - 1) continue;
      if (it_circ == n_states - 1) {
        for (unsigned i_state = 0; i_state < n_states; i_state++) {
          v[i_state] = w[i_state];
          Hv[i_state] = Hw[i_state];
        }
        for (unsigned i_state = 0; i_state < n_states; i_state++) {
          lowest_eigenvalues[i_state] = Util::dot_omp(v[i_state], Hv[i_state]);
          h_krylov(i_state, i_state) = lowest_eigenvalues[i_state];
          for (unsigned k_state = i_state + 1; k_state < n_states; k_state++) {
            double element = Util::dot_omp(v[i_state], Hv[k_state]);
            h_krylov(i_state, k_state) = element;
            h_krylov(k_state, i_state) = element;
          }
        }
        continue;
      }
    }

#pragma omp parallel for
    for (size_t j = 0; j < dim; j++) {
      const double diff_to_diag = lowest_eigenvalues[i_state_precond] - matrix.get_diag(j);  // diag_elems[j];
      if (std::abs(diff_to_diag) < 1.0e-8) {
        v[it_circ][j] = 0.;
      } else {
        v[it_circ][j] = (Hw[i_state_precond][j] - lowest_eigenvalues[i_state_precond] * w[i_state_precond][j]) / diff_to_diag;
      }
    }

    // Orthogonalize and normalize.
    for (size_t i = 0; i < it_circ; i++) {
      double norm = Util::dot_omp(v[it_circ], v[i]);
#pragma omp parallel for
      for (size_t j = 0; j < dim; j++) {
        v[it_circ][j] -= norm * v[i][j];
      }
    }
    double norm = sqrt(Util::dot_omp(v[it_circ], v[it_circ]));
    if (norm<1e-12) { // corner case: norm gets small before eigenvalues converge
      converged = true;
      break;
    }

#pragma omp parallel for
    for (size_t j = 0; j < dim; j++) {
      v[it_circ][j] /= norm;
    }

    Hv[it_circ] = matrix.mul(v[it_circ]);

    // Construct subspace matrix.
    for (size_t i = 0; i <= it_circ; i++) {
      h_krylov(it_circ, i) = Util::dot_omp(v[i], Hv[it_circ]);
      //h_krylov(it_circ, i) = h_krylov(i, it_circ); // only lower trianguluar part is referenced
    }

    // Diagonalize subspace matrix.
    if (i_state_precond + 1 == n_states) {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(
          h_krylov.leftCols(it_circ + 1).topRows(it_circ + 1));
      const auto& eigenvals = eigenSolver.eigenvalues();  // in ascending order
      const auto& eigenvecs = eigenSolver.eigenvectors();
      for (unsigned i_state = 0; i_state < n_states; i_state++) {
        lowest_eigenvalues[i_state] = eigenvals(i_state);
        double factor = 1.0;
        if (eigenvecs(0, i_state) < 0) factor = -1.0;
        for (size_t i = 0; i < it_circ + 1; i++)
          eigenvector_krylov(i, i_state) = eigenvecs(i, i_state) * factor;
#pragma omp parallel for
        for (size_t j = 0; j < dim; j++) {
          double w_j = 0.0;
          double Hw_j = 0.0;
          for (size_t i = 0; i < it_circ + 1; i++) {
            w_j += v[i][j] * eigenvector_krylov(i, i_state);
            Hw_j += Hv[i][j] * eigenvector_krylov(i, i_state);
          }
          w[i_state][j] = w_j;
          Hw[i_state][j] = Hw_j;
        }
      }

      if (verbose) {
        printf("Davidson #%zu:", it_real);
        for (const auto& eigenval : lowest_eigenvalues) printf("  %.10f", eigenval);
        printf("\n");
      }
      it_real++;
      for (unsigned i_state = n_converged; i_state < n_states; i_state++) {
        if (std::abs(lowest_eigenvalues[i_state] - lowest_eigenvalues_prev[i_state]) > TOLERANCE) {
          break;
        } else {
          n_converged++;
        }
      }
      if (n_converged == n_states) converged = true;

      if (!converged) lowest_eigenvalues_prev = lowest_eigenvalues;

      if (converged) break;
    }
    i_state_precond++;
    if (i_state_precond >= n_states) i_state_precond = n_converged;
  }
  lowest_eigenvectors = w;
  if (n_states < initial_vectors.size()) { // Corner case for excited states
    lowest_eigenvectors.resize(initial_vectors.size());
    for (unsigned i = n_states; i < initial_vectors.size(); i++) 
      lowest_eigenvectors[i] = initial_vectors[i];
  }
}
