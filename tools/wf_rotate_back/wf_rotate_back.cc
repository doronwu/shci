#include <hps/src/hps.h>
#include <shci/src/det/det.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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

static double minor_det(const std::vector<std::vector<double>>& rot,
                        const std::vector<unsigned>& old_occ,
                        const std::vector<unsigned>& new_occ) {
  const size_t n = old_occ.size();
  Eigen::MatrixXd sub(n, n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      sub(i, j) = rot[old_occ[i]][new_occ[j]];
    }
  }
  return sub.determinant();
}

static std::string occ_to_key(const std::vector<unsigned>& up, const std::vector<unsigned>& dn) {
  std::ostringstream os;
  for (unsigned x : up) os << x << ',';
  os << '|';
  for (unsigned x : dn) os << x << ',';
  return os.str();
}

int main(int argc, char* argv[]) {
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

  std::unordered_map<std::string, double> old_coef_map;
  old_coef_map.reserve(alpha_combs.size() * 2);

  for (const auto& e : ci_new) {
    std::vector<std::pair<const std::vector<unsigned>*, double>> alpha_terms;
    alpha_terms.reserve(alpha_combs.size());
    for (const auto& occ : alpha_combs) {
      const double a = minor_det(rot, occ.occ, e.up_occ);
      if (std::abs(a) >= amp_cut) alpha_terms.push_back({&occ.occ, a});
    }

    std::vector<std::pair<const std::vector<unsigned>*, double>> beta_terms;
    beta_terms.reserve(beta_combs.size());
    for (const auto& occ : beta_combs) {
      const double b = minor_det(rot, occ.occ, e.dn_occ);
      if (std::abs(b) >= amp_cut) beta_terms.push_back({&occ.occ, b});
    }

    for (const auto& a : alpha_terms) {
      for (const auto& b : beta_terms) {
        const double contrib = e.coef * a.second * b.second;
        if (std::abs(contrib) < amp_cut) continue;
        old_coef_map[occ_to_key(*a.first, *b.first)] += contrib;
      }
    }
  }

  std::vector<CIEntry> ci_old;
  ci_old.reserve(old_coef_map.size());
  for (const auto& kv : old_coef_map) {
    std::vector<unsigned> up;
    std::vector<unsigned> dn;
    bool dn_part = false;
    unsigned cur = 0;
    bool in_num = false;
    for (char ch : kv.first) {
      if (ch == '|') {
        if (in_num) {
          up.push_back(cur);
          cur = 0;
          in_num = false;
        }
        dn_part = true;
      } else if (ch == ',') {
        if (in_num) {
          if (!dn_part) up.push_back(cur);
          else dn.push_back(cur);
          cur = 0;
          in_num = false;
        }
      } else if (ch >= '0' && ch <= '9') {
        cur = cur * 10 + static_cast<unsigned>(ch - '0');
        in_num = true;
      }
    }
    ci_old.push_back({up, dn, kv.second});
  }

  std::sort(ci_old.begin(), ci_old.end(), [](const CIEntry& a, const CIEntry& b) {
    return std::abs(a.coef) > std::abs(b.coef);
  });

  {
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

  {
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

  std::cout << "Loaded wf: n_dets=" << wf.get_n_dets() << ", n_states=" << wf.coefs.size() << "\n";
  std::cout << "Using state=" << i_state << ", selected new-basis CI count=" << ci_new.size() << "\n";
  std::cout << "Generated old-basis CI count=" << ci_old.size() << "\n";
  std::cout << "Output: " << out_prefix << "_new_basis.tsv and " << out_prefix << "_old_basis.tsv\n";
  return 0;
}
