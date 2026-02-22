#include <hps/src/hps.h>
#include <shci/src/det/det.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef SHCI_USE_MPI
#include <mpi.h>
#endif

#ifdef SHCI_USE_MPI
class MpiGuard {
 public:
  MpiGuard(int* argc, char*** argv) { MPI_Init(argc, argv); }
  ~MpiGuard() { MPI_Finalize(); }
};
#endif

constexpr double SQRT2_INV = 0.7071067811865475;

class Wavefunction {
 public:
  unsigned n_up = 0;
  unsigned n_dn = 0;
  double energy_hf = 0.0;
  std::vector<double> energy_var;
  bool time_sym = false;
  std::vector<Det> dets;
  std::vector<std::vector<double>> coefs;

  size_t get_n_dets() const { return dets.size(); }

  void unpack_time_sym() {
    const size_t n_dets_old = get_n_dets();
    for (size_t i = 0; i < n_dets_old; i++) {
      const auto& det = dets[i];
      if (det.up < det.dn) {
        Det det_rev = det;
        det_rev.reverse_spin();
        for (auto& state_coefs : coefs) {
          const double coef_new = state_coefs[i] * SQRT2_INV;
          state_coefs[i] = coef_new;
          state_coefs.push_back(coef_new);
        }
        dets.push_back(det_rev);
      }
    }
  }

  template <class B>
  void parse(B& buf) {
    buf >> n_up >> n_dn >> dets >> coefs >> energy_hf >> energy_var >> time_sym;
    if (time_sym) unpack_time_sym();
  }
};

struct CIEntry {
  std::vector<unsigned> up_occ;
  std::vector<unsigned> dn_occ;
  double coef = 0.0;
};

struct OrbitalComb {
  std::vector<unsigned> occ;
};

struct VecHash {
  size_t operator()(const std::vector<unsigned>& v) const {
    size_t seed = v.size();
    for (unsigned x : v) {
      seed ^= static_cast<size_t>(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

struct DetKey {
  std::vector<unsigned> up;
  std::vector<unsigned> dn;

  bool operator==(const DetKey& other) const {
    return up == other.up && dn == other.dn;
  }
};

struct DetKeyHash {
  size_t operator()(const DetKey& key) const {
    size_t seed = VecHash{}(key.up);
    seed ^= VecHash{}(key.dn) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

static void merge_old_coef_maps(std::unordered_map<DetKey, double, DetKeyHash>& dst,
                               const std::unordered_map<DetKey, double, DetKeyHash>& src) {
  for (const auto& kv : src) dst[kv.first] += kv.second;
}

static std::string serialize_old_coef_map(const std::unordered_map<DetKey, double, DetKeyHash>& m) {
  std::ostringstream os(std::ios::binary);
  const uint64_t n = static_cast<uint64_t>(m.size());
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  for (const auto& kv : m) {
    const uint64_t nu = static_cast<uint64_t>(kv.first.up.size());
    const uint64_t nd = static_cast<uint64_t>(kv.first.dn.size());
    os.write(reinterpret_cast<const char*>(&nu), sizeof(nu));
    os.write(reinterpret_cast<const char*>(kv.first.up.data()), static_cast<std::streamsize>(nu * sizeof(unsigned)));
    os.write(reinterpret_cast<const char*>(&nd), sizeof(nd));
    os.write(reinterpret_cast<const char*>(kv.first.dn.data()), static_cast<std::streamsize>(nd * sizeof(unsigned)));
    os.write(reinterpret_cast<const char*>(&kv.second), sizeof(kv.second));
  }
  return os.str();
}

static void deserialize_into_old_coef_map(const char* data, size_t size,
                                          std::unordered_map<DetKey, double, DetKeyHash>& out) {
  std::string blob(data, size);
  std::istringstream is(blob, std::ios::binary);
  uint64_t n = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof(n));
  for (uint64_t i = 0; i < n; ++i) {
    uint64_t nu = 0;
    uint64_t nd = 0;
    DetKey key;
    is.read(reinterpret_cast<char*>(&nu), sizeof(nu));
    key.up.resize(static_cast<size_t>(nu));
    is.read(reinterpret_cast<char*>(key.up.data()), static_cast<std::streamsize>(nu * sizeof(unsigned)));
    is.read(reinterpret_cast<char*>(&nd), sizeof(nd));
    key.dn.resize(static_cast<size_t>(nd));
    is.read(reinterpret_cast<char*>(key.dn.data()), static_cast<std::streamsize>(nd * sizeof(unsigned)));
    double coef = 0.0;
    is.read(reinterpret_cast<char*>(&coef), sizeof(coef));
    out[key] += coef;
  }
}

static void usage() {
  std::cout << "Usage:\n"
            << "  wf_rotate_back <wf.dat> <rot_matrix.txt> [options]\n\n"
            << "Options:\n"
            << "  --state <i>          CI state index, default 0\n"
            << "  --top-new <n>        keep top |CI| determinants from wf, default all\n"
            << "  --ci-cut <x>         ignore |CI| < x in new basis, default 0\n"
            << "  --amp-cut <x>        ignore minor amplitudes < x in back-transform, default 1e-12\n"
            << "  --max-comb <n>       max C(norb, nelec) per spin allowed, default 200000\n"
            << "  --out-prefix <p>     output prefix, default ci_rotated\n";
}

static bool starts_with(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

static std::vector<std::vector<double>> load_rotation_matrix(const std::string& filename) {
  std::ifstream in(filename);
  if (!in) throw std::runtime_error("Failed to open rotation matrix file: " + filename);

  std::vector<std::vector<double>> mat;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (starts_with(line, "#")) continue;
    std::stringstream ss(line);
    std::vector<double> row;
    double x = 0.0;
    while (ss >> x) row.push_back(x);
    if (!row.empty()) mat.push_back(row);
  }

  if (mat.empty()) throw std::runtime_error("Rotation matrix file has no numeric data");

  const size_t n = mat[0].size();
  for (const auto& row : mat) {
    if (row.size() != n) {
      throw std::runtime_error("Rotation matrix rows have inconsistent lengths");
    }
  }
  if (mat.size() != n) {
    throw std::runtime_error("Rotation matrix must be square");
  }
  return mat;
}

static double comb_count(unsigned n, unsigned k) {
  if (k > n) return 0.0;
  if (k == 0 || k == n) return 1.0;
  k = std::min(k, n - k);
  long double res = 1.0;
  for (unsigned i = 1; i <= k; ++i) {
    res = res * (n - k + i) / i;
  }
  return static_cast<double>(res);
}

static void enumerate_combs_rec(unsigned n, unsigned k, unsigned start,
                                std::vector<unsigned>& cur,
                                std::vector<OrbitalComb>& out) {
  if (cur.size() == k) {
    out.push_back({cur});
    return;
  }
  for (unsigned i = start; i < n; ++i) {
    cur.push_back(i);
    enumerate_combs_rec(n, k, i + 1, cur, out);
    cur.pop_back();
  }
}

static std::vector<OrbitalComb> enumerate_combs(unsigned n, unsigned k) {
  std::vector<OrbitalComb> out;
  std::vector<unsigned> cur;
  enumerate_combs_rec(n, k, 0, cur, out);
  return out;
}

static double minor_det_inplace(const std::vector<std::vector<double>>& rot,
                                const std::vector<unsigned>& old_occ,
                                const std::vector<unsigned>& new_occ,
                                std::vector<double>& sub,
                                std::vector<size_t>& pivots) {
  const size_t n = old_occ.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      sub[i * n + j] = rot[old_occ[i]][new_occ[j]];
    }
  }

  double det_sign = 1.0;
  for (size_t k = 0; k < n; ++k) {
    size_t pivot = k;
    double max_abs = std::abs(sub[k * n + k]);
    for (size_t i = k + 1; i < n; ++i) {
      const double v = std::abs(sub[i * n + k]);
      if (v > max_abs) {
        max_abs = v;
        pivot = i;
      }
    }
    pivots[k] = pivot;
    if (max_abs == 0.0) return 0.0;

    if (pivot != k) {
      for (size_t j = 0; j < n; ++j) {
        std::swap(sub[k * n + j], sub[pivot * n + j]);
      }
      det_sign = -det_sign;
    }

    const double pivot_val = sub[k * n + k];
    for (size_t i = k + 1; i < n; ++i) {
      const double factor = sub[i * n + k] / pivot_val;
      sub[i * n + k] = factor;
      for (size_t j = k + 1; j < n; ++j) {
        sub[i * n + j] -= factor * sub[k * n + j];
      }
    }
  }

  double det = det_sign;
  for (size_t i = 0; i < n; ++i) det *= sub[i * n + i];
  return det;
}

int main(int argc, char* argv[]) {
#ifdef SHCI_USE_MPI
  MpiGuard mpi_guard(&argc, &argv);
  int mpi_rank = 0;
  int mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#else
  const int mpi_rank = 0;
  const int mpi_size = 1;
#endif

  if (argc < 3) {
    usage();
    return 1;
  }

  const std::string wf_file = argv[1];
  const std::string rot_file = argv[2];

  size_t i_state = 0;
  size_t top_new = 0;
  double ci_cut = 0.0;
  double amp_cut = 1e-12;
  size_t max_comb = 200000;
  std::string out_prefix = "ci_rotated";

  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const std::string& name) {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
    };

    if (arg == "--state") {
      need_value(arg);
      i_state = std::stoull(argv[++i]);
    } else if (arg == "--top-new") {
      need_value(arg);
      top_new = std::stoull(argv[++i]);
    } else if (arg == "--ci-cut") {
      need_value(arg);
      ci_cut = std::stod(argv[++i]);
    } else if (arg == "--amp-cut") {
      need_value(arg);
      amp_cut = std::stod(argv[++i]);
    } else if (arg == "--max-comb") {
      need_value(arg);
      max_comb = std::stoull(argv[++i]);
    } else if (arg == "--out-prefix") {
      need_value(arg);
      out_prefix = argv[++i];
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  std::ifstream serialized_wf(wf_file, std::ios::binary);
  if (!serialized_wf) throw std::runtime_error("Failed to open wf file: " + wf_file);
  Wavefunction wf = hps::from_stream<Wavefunction>(serialized_wf);

  if (i_state >= wf.coefs.size()) {
    throw std::runtime_error("state index out of range");
  }

  const auto rot = load_rotation_matrix(rot_file);
  const unsigned n_orb = static_cast<unsigned>(rot.size());

  if (wf.n_up > n_orb || wf.n_dn > n_orb) {
    throw std::runtime_error("rotation matrix size smaller than electron count");
  }

  const double n_alpha_comb = comb_count(n_orb, wf.n_up);
  const double n_beta_comb = comb_count(n_orb, wf.n_dn);
  if (n_alpha_comb > static_cast<double>(max_comb) || n_beta_comb > static_cast<double>(max_comb)) {
    std::ostringstream os;
    os << "Combination count too large: C(" << n_orb << "," << wf.n_up << ")=" << n_alpha_comb
       << ", C(" << n_orb << "," << wf.n_dn << ")=" << n_beta_comb
       << ". Increase --max-comb if you really want this.";
    throw std::runtime_error(os.str());
  }

  std::vector<size_t> inds(wf.dets.size());
  std::iota(inds.begin(), inds.end(), 0);
  std::sort(inds.begin(), inds.end(), [&](size_t a, size_t b) {
    return std::abs(wf.coefs[i_state][a]) > std::abs(wf.coefs[i_state][b]);
  });

  std::vector<CIEntry> ci_new;
  ci_new.reserve(inds.size());
  for (size_t idx : inds) {
    const double c = wf.coefs[i_state][idx];
    if (std::abs(c) < ci_cut) continue;
    const auto& det = wf.dets[idx];
    ci_new.push_back({det.up.get_occupied_orbs(), det.dn.get_occupied_orbs(), c});
    if (top_new > 0 && ci_new.size() >= top_new) break;
  }

  const auto alpha_combs = enumerate_combs(n_orb, wf.n_up);
  const auto beta_combs = enumerate_combs(n_orb, wf.n_dn);

  std::unordered_map<std::vector<unsigned>, size_t, VecHash> up_occ_id_map;
  std::unordered_map<std::vector<unsigned>, size_t, VecHash> dn_occ_id_map;
  up_occ_id_map.reserve(ci_new.size());
  dn_occ_id_map.reserve(ci_new.size());

  std::vector<size_t> ci_up_ids(ci_new.size(), 0);
  std::vector<size_t> ci_dn_ids(ci_new.size(), 0);
  std::vector<const std::vector<unsigned>*> unique_up_occs;
  std::vector<const std::vector<unsigned>*> unique_dn_occs;
  unique_up_occs.reserve(ci_new.size());
  unique_dn_occs.reserve(ci_new.size());

  for (size_t i = 0; i < ci_new.size(); ++i) {
    const auto& e = ci_new[i];
    const auto up_insert = up_occ_id_map.emplace(e.up_occ, unique_up_occs.size());
    if (up_insert.second) unique_up_occs.push_back(&up_insert.first->first);
    ci_up_ids[i] = up_insert.first->second;

    const auto dn_insert = dn_occ_id_map.emplace(e.dn_occ, unique_dn_occs.size());
    if (dn_insert.second) unique_dn_occs.push_back(&dn_insert.first->first);
    ci_dn_ids[i] = dn_insert.first->second;
  }

  std::unordered_map<DetKey, double, DetKeyHash> old_coef_map;
  old_coef_map.reserve(alpha_combs.size() * 2);

  const bool use_parallel = ci_new.size() > 1;
  const bool has_reuse = unique_up_occs.size() < ci_new.size() || unique_dn_occs.size() < ci_new.size();

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = use_parallel ? omp_get_max_threads() : 1;
#endif
  std::vector<std::unordered_map<DetKey, double, DetKeyHash>> old_coef_maps_local(n_threads);
  for (auto& m : old_coef_maps_local) m.reserve(alpha_combs.size());

  if (has_reuse) {
    std::vector<std::vector<std::pair<size_t, double>>> alpha_terms_by_new_occ(unique_up_occs.size());
    std::vector<std::vector<std::pair<size_t, double>>> beta_terms_by_new_occ(unique_dn_occs.size());

#pragma omp parallel if(use_parallel)
    {
      std::vector<double> alpha_sub(static_cast<size_t>(wf.n_up) * wf.n_up, 0.0);
      std::vector<size_t> alpha_pivots(wf.n_up, 0);

#pragma omp for schedule(dynamic)
      for (size_t new_id = 0; new_id < unique_up_occs.size(); ++new_id) {
        auto& alpha_terms = alpha_terms_by_new_occ[new_id];
        alpha_terms.reserve(alpha_combs.size());
        const auto& new_occ = *unique_up_occs[new_id];
        for (size_t old_id = 0; old_id < alpha_combs.size(); ++old_id) {
          const auto& old_occ = alpha_combs[old_id].occ;
          const double a = minor_det_inplace(rot, old_occ, new_occ, alpha_sub, alpha_pivots);
          if (std::abs(a) >= amp_cut) alpha_terms.push_back({old_id, a});
        }
      }

      std::vector<double> beta_sub(static_cast<size_t>(wf.n_dn) * wf.n_dn, 0.0);
      std::vector<size_t> beta_pivots(wf.n_dn, 0);

#pragma omp for schedule(dynamic)
      for (size_t new_id = 0; new_id < unique_dn_occs.size(); ++new_id) {
        auto& beta_terms = beta_terms_by_new_occ[new_id];
        beta_terms.reserve(beta_combs.size());
        const auto& new_occ = *unique_dn_occs[new_id];
        for (size_t old_id = 0; old_id < beta_combs.size(); ++old_id) {
          const auto& old_occ = beta_combs[old_id].occ;
          const double b = minor_det_inplace(rot, old_occ, new_occ, beta_sub, beta_pivots);
          if (std::abs(b) >= amp_cut) beta_terms.push_back({old_id, b});
        }
      }
    }

#pragma omp parallel if(use_parallel)
    {
#ifdef _OPENMP
      auto& old_coef_map_local = old_coef_maps_local[omp_get_thread_num()];
#else
      auto& old_coef_map_local = old_coef_maps_local[0];
#endif

#pragma omp for schedule(dynamic)
      for (size_t i = static_cast<size_t>(mpi_rank); i < ci_new.size(); i += static_cast<size_t>(mpi_size)) {
        const auto& e = ci_new[i];
        const auto& alpha_terms = alpha_terms_by_new_occ[ci_up_ids[i]];
        const auto& beta_terms = beta_terms_by_new_occ[ci_dn_ids[i]];

        for (const auto& a : alpha_terms) {
          for (const auto& b : beta_terms) {
            const double contrib = e.coef * a.second * b.second;
            if (std::abs(contrib) < amp_cut) continue;
            old_coef_map_local[{alpha_combs[a.first].occ, beta_combs[b.first].occ}] += contrib;
          }
        }
      }
    }
  } else {
#pragma omp parallel if(use_parallel)
    {
#ifdef _OPENMP
      auto& old_coef_map_local = old_coef_maps_local[omp_get_thread_num()];
#else
      auto& old_coef_map_local = old_coef_maps_local[0];
#endif

      std::vector<double> alpha_sub(static_cast<size_t>(wf.n_up) * wf.n_up, 0.0);
      std::vector<size_t> alpha_pivots(wf.n_up, 0);
      std::vector<double> beta_sub(static_cast<size_t>(wf.n_dn) * wf.n_dn, 0.0);
      std::vector<size_t> beta_pivots(wf.n_dn, 0);

#pragma omp for schedule(dynamic)
      for (size_t i = static_cast<size_t>(mpi_rank); i < ci_new.size(); i += static_cast<size_t>(mpi_size)) {
        const auto& e = ci_new[i];
        std::vector<std::pair<size_t, double>> alpha_terms;
        alpha_terms.reserve(alpha_combs.size());
        for (size_t old_id = 0; old_id < alpha_combs.size(); ++old_id) {
          const auto& old_occ = alpha_combs[old_id].occ;
          const double a = minor_det_inplace(rot, old_occ, e.up_occ, alpha_sub, alpha_pivots);
          if (std::abs(a) >= amp_cut) alpha_terms.push_back({old_id, a});
        }

        std::vector<std::pair<size_t, double>> beta_terms;
        beta_terms.reserve(beta_combs.size());
        for (size_t old_id = 0; old_id < beta_combs.size(); ++old_id) {
          const auto& old_occ = beta_combs[old_id].occ;
          const double b = minor_det_inplace(rot, old_occ, e.dn_occ, beta_sub, beta_pivots);
          if (std::abs(b) >= amp_cut) beta_terms.push_back({old_id, b});
        }

        for (const auto& a : alpha_terms) {
          for (const auto& b : beta_terms) {
            const double contrib = e.coef * a.second * b.second;
            if (std::abs(contrib) < amp_cut) continue;
            old_coef_map_local[{alpha_combs[a.first].occ, beta_combs[b.first].occ}] += contrib;
          }
        }
      }
    }
  }

  for (const auto& old_coef_map_local : old_coef_maps_local) {
    merge_old_coef_maps(old_coef_map, old_coef_map_local);
  }

#ifdef SHCI_USE_MPI
  if (mpi_size > 1) {
    const std::string payload = serialize_old_coef_map(old_coef_map);
    const int local_size = static_cast<int>(payload.size());
    std::vector<int> recv_sizes;
    if (mpi_rank == 0) recv_sizes.resize(mpi_size, 0);
    int* recv_sizes_ptr = mpi_rank == 0 ? recv_sizes.data() : nullptr;
    MPI_Gather(&local_size, 1, MPI_INT, recv_sizes_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs;
    std::vector<char> recv_buf;
    if (mpi_rank == 0) {
      displs.resize(mpi_size, 0);
      int total = 0;
      for (int r = 0; r < mpi_size; ++r) {
        displs[r] = total;
        total += recv_sizes[r];
      }
      recv_buf.resize(total);
    }

    char* recv_buf_ptr = mpi_rank == 0 ? recv_buf.data() : nullptr;
    int* displs_ptr = mpi_rank == 0 ? displs.data() : nullptr;
    MPI_Gatherv(payload.data(), local_size, MPI_CHAR,
                recv_buf_ptr, recv_sizes_ptr, displs_ptr, MPI_CHAR,
                0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
      std::unordered_map<DetKey, double, DetKeyHash> merged;
      merged.reserve(old_coef_map.size() * static_cast<size_t>(mpi_size));
      for (int r = 0; r < mpi_size; ++r) {
        deserialize_into_old_coef_map(recv_buf.data() + displs[r], static_cast<size_t>(recv_sizes[r]), merged);
      }
      old_coef_map.swap(merged);
    } else {
      old_coef_map.clear();
    }
  }
#endif

  std::vector<CIEntry> ci_old;
  ci_old.reserve(old_coef_map.size());
  for (const auto& kv : old_coef_map) {
    ci_old.push_back({kv.first.up, kv.first.dn, kv.second});
  }

  std::sort(ci_old.begin(), ci_old.end(), [](const CIEntry& a, const CIEntry& b) {
    return std::abs(a.coef) > std::abs(b.coef);
  });

  if (mpi_rank == 0) {
    std::ofstream out(out_prefix + "_new_basis.tsv");
    out << "# up_occ(1-based)\tdn_occ(1-based)\tcoef\n";
    out << std::setprecision(16);
    for (const auto& e : ci_new) {
      for (unsigned x : e.up_occ) out << (x + 1) << ' ';
      out << '\t';
      for (unsigned x : e.dn_occ) out << (x + 1) << ' ';
      out << '\t' << e.coef << '\n';
    }
  }

  if (mpi_rank == 0) {
    std::ofstream out(out_prefix + "_old_basis.tsv");
    out << "# up_occ(1-based)\tdn_occ(1-based)\tcoef\n";
    out << std::setprecision(16);
    for (const auto& e : ci_old) {
      for (unsigned x : e.up_occ) out << (x + 1) << ' ';
      out << '\t';
      for (unsigned x : e.dn_occ) out << (x + 1) << ' ';
      out << '\t' << e.coef << '\n';
    }
  }

  if (mpi_rank == 0) {
    std::cout << "Loaded wf: n_dets=" << wf.get_n_dets() << ", n_states=" << wf.coefs.size() << "\n";
    std::cout << "Using state=" << i_state << ", selected new-basis CI count=" << ci_new.size() << "\n";
    std::cout << "Generated old-basis CI count=" << ci_old.size() << "\n";
    std::cout << "Output: " << out_prefix << "_new_basis.tsv and " << out_prefix << "_old_basis.tsv\n";
  }
  return 0;
}
