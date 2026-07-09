#include <vector>
#include <algorithm>
#include <numeric>   // iota
#include <cstdio>    // printf
#include <cmath>     // std::abs
#include <stdexcept> // runtime_error
#include <shci/src/det/det.h>

// Assumed minimal interfaces:
//
// struct BitDet {
//     std::vector<unsigned> get_occupied_orbs() const;
// };
//
// struct Det {
//     BitDet up, dn;
// };
//
// struct Wavefunction {
//     // coefs[state][det_index]
//     std::vector<std::vector<double>> coefs;
//     std::vector<Det> dets;
// };

void print_sorted_determinants(
    const Wavefunction& wf,
    std::size_t i_state,
    double threshold,
    bool norb_provided,
    const std::vector<unsigned>& orbs // used when norb_provided == true
) {
    // --- basic validation ---
    if (i_state >= wf.coefs.size()) {
        throw std::runtime_error("i_state out of range for wf.coefs");
    }
    const auto& coefs = wf.coefs[i_state];
    if (coefs.size() != wf.dets.size()) {
        throw std::runtime_error("Size mismatch: wf.coefs[i_state].size() != wf.dets.size()");
    }

    // indices 0..n_det-1
    std::vector<std::size_t> inds(coefs.size());
    std::iota(inds.begin(), inds.end(), 0);

    // sort by |coef| descending
    std::sort(inds.begin(), inds.end(),
              [&](const std::size_t a, const std::size_t b) {
                  return std::abs(coefs[a]) > std::abs(coefs[b]);
              });

    // determine cutoff by threshold (first |coef| < threshold), or keep all
    std::size_t N_cutoff = inds.size();

    for (std::size_t i = 0; i < inds.size(); ++i) {
        const double mag = std::abs(coefs[inds[i]]);
        if (mag < threshold) {
            N_cutoff = i;
            break;
        }
    }


    // print each determinant up to cutoff
    for (std::size_t j = 0; j < N_cutoff; ++j) {
        const std::size_t det_idx = inds[j];
        const Det& det = wf.dets[det_idx];

        // coefficient
        std::printf("%16.3e", coefs[det_idx]);

        // bit-format using the provided orbital list
        if (norb_provided) {
            std::printf("|");
            const auto up_orbs = det.up.get_occupied_orbs();
            for (unsigned i_orb : orbs) {
                const bool occ = std::find(up_orbs.begin(), up_orbs.end(), i_orb) != up_orbs.end();
                std::printf("%c", occ ? '1' : '0');
            }
            std::printf(">|");

            const auto dn_orbs = det.dn.get_occupied_orbs();
            for (unsigned i_orb : orbs) {
                const bool occ = std::find(dn_orbs.begin(), dn_orbs.end(), i_orb) != dn_orbs.end();
                std::printf("%c", occ ? '1' : '0');
            }
            std::printf(">\n");
        }
        // occupied-orbital list format
        else {
            const auto up_orbs = det.up.get_occupied_orbs();
            const auto dn_orbs = det.dn.get_occupied_orbs();

            std::printf("|");
            for (unsigned orb : up_orbs) std::printf("%u ", orb);
            std::printf(">|");
            for (unsigned orb : dn_orbs) std::printf("%u ", orb);
            std::printf(">\n");
        }
    }
}
