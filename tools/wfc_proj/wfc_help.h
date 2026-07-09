#include <vector>
#include <algorithm>
#include <numeric>   // iota
#include <cstdio>    // printf
#include <cmath>     // std::abs
#include <stdexcept> // runtime_error
#include <shci/src/det/det.h>

//The wavefunction clss 
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
        for (auto& state_coefs: coefs) {                                                                                                                     
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

// helper, print a list of orbitals
template <typename T>
void print_orbs(const std::string& name, const std::vector<T>& orbs) {
    std::cout << name << " = [ ";
    for (size_t i = 0; i < orbs.size(); ++i) {
        std::cout << orbs[i];
        if (i + 1 < orbs.size()) std::cout << ", ";
    }
    std::cout << " ]\n";
}

//method: print the sorted wavefunction with certain threshold
void print_sorted_determinants(
    const Wavefunction& wf, //the wavefunction object
    std::size_t i_state, // the index of state
    double threshold, //the threshold for coefficient
    bool norb_provided, //if true, use the string format for orbital; if false, use the occupation format
    const std::vector<unsigned>& orbs // used when norb_provided == true, the list of orbitals, (use the 0,1,..norbs index)
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
        std::printf("%+16.3e", coefs[det_idx]);

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

//method: print the sorted wavefunction within the Gs
void print_sorted_determinants_in_Gs(
    const Wavefunction& wf, //the wavefunction object
    std::size_t i_state, // the index of state
    double threshold, //the threshold for coefficient
    bool norb_provided, //if true, use the string format for orbital; if false, use the occupation format
    const std::vector<unsigned>& orbs, // used when norb_provided == true, the list of orbitals, (use the 0,1,..norbs index)
    std::vector<size_t> Gs_index_map
) {
    // --- basic validation ---
    if (i_state >= wf.coefs.size()) {
        throw std::runtime_error("i_state out of range for wf.coefs");
    }
    const auto& coefs = wf.coefs[i_state];
    if (coefs.size() != wf.dets.size()) {
        throw std::runtime_error("Size mismatch: wf.coefs[i_state].size() != wf.dets.size()");
    }

    // indices 0..n_det-1 replaced with Gs_index_map

    // sort by |coef| descending
    std::sort(Gs_index_map.begin(), Gs_index_map.end(),
              [&](const std::size_t a, const std::size_t b) {
                  return std::abs(coefs[a]) > std::abs(coefs[b]);
              });
    // determine cutoff by threshold (first |coef| < threshold), or keep all
    std::size_t N_cutoff = Gs_index_map.size();

    for (std::size_t i = 0; i < Gs_index_map.size(); ++i) {
        const double mag = std::abs(coefs[Gs_index_map[i]]);
        if (mag < threshold) {
            N_cutoff = i;
            break;
        }
    }


    // print each determinant up to cutoff
    for (std::size_t j = 0; j < N_cutoff; ++j) {
        const std::size_t det_idx = Gs_index_map[j];
        const Det& det = wf.dets[det_idx];

        // coefficient
        std::printf("%+16.3e", coefs[det_idx]);

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

//get the index map of finding index of band in small active space in large active space:
std::vector<std::size_t> map_indices(
    const std::vector<int>& short_list,
    const std::vector<int>& long_list
) {
    // Build lookup map: value -> index in long_list
    std::unordered_map<int, std::size_t> index_map;
    index_map.reserve(long_list.size());

    for (std::size_t i = 0; i < long_list.size(); ++i) {
        index_map[long_list[i]] = i; // key: band number, value: its index
    }

    // Pre-size result vector to short_list length
    std::vector<std::size_t> result(short_list.size());

    // Fill result in place
    for (std::size_t j = 0; j < short_list.size(); ++j) {
        int val = short_list[j]; // band number in short list
        auto it = index_map.find(val); //find key band number,
        // <it> is the (bandnumber, index) key-value pair if found, else it is index_map.end()
        if (it != index_map.end()) {
            result[j] = it->second; //it->first is the key, it->second is the value
        } else {
            throw std::runtime_error("Band " + std::to_string(val) +
                                     " not found in large active space, bad case! ");
        }
    }
    // the result is a small active space size list; where result[index in A_S] = (index in A_L)
    return result;
}

//Compute lambda:

struct LambdaResult {
    double Lambda;                   // Use double for accuracy
    std::vector<size_t> Gs_index_map; // Indices of determinants in G_S
};

LambdaResult Compute_Lambda(
    const std::vector<size_t>& index_map,
    size_t i_state_l,
    const Wavefunction& wf_l,
    const int& n_up_s, 
    const int& n_dn_s,
    const bool& Verbose
) {
    const size_t& n_det = wf_l.dets.size(); // number of determinants
    std::vector<int> Ne_S_up(n_det); // number of electrons in in small active space subspace, spin up component
    std::vector<int> Ne_S_dn(n_det); // number of electrons in in small active space subspace, spin dn component

    size_t AS_first = index_map.front();   // first band in A_S
    size_t AS_last  = index_map.back();    // last band in A_S
    //reference occupied list: 
    std::vector<unsigned> reference_occupied_orbitals;
    reference_occupied_orbitals.reserve(AS_first);  // optional, for efficiency
    for (unsigned i = 0; i < AS_first; ++i) {
        reference_occupied_orbitals.push_back(i);
    }
    double Lambda = 0.0f;
    //
    int count_Gs = 0; // number of determinants with projection on determinant of small active space
    std::vector<size_t> Gs_index_map; // the list of index of determinant that have no transfer from A_S to Res orbitals
    int count_e_transter_outside = 0; // the number of electrons transfered within the A_L-A_S
    int count_1e_transfer = 0; // the case when 1 electron has exchanged with outside of A_S
    int count_2e_transfer = 0;// the case when 2 electron has exchanged with outside of A_S
    int count_3e_transfer = 0;// the case when 3 electron has exchanged with outside of A_S
    int count_4e_transfer = 0;// the case when 4 electron has exchanged with outside of A_S
    int count_higher_transfer = 0; //the case when more elctrons has exchanged with outside of A_S 

    for (size_t j = 0; j < n_det; ++j){ // loop through all determinants
        int ne_s_up = 0; // number of electrons in A_S subspace, up
        int ne_s_dn = 0; // number of electrons in A_S subspace, dn
        const Det &det = wf_l.dets[j];
        std::vector<int> overlap_list_up; // the list of overlaped band between A_S and A_L
        std::vector<int> overlap_list_dn;
        std::vector<unsigned> up_orbs(det.up.get_occupied_orbs());
        std::vector<unsigned> dn_orbs(det.dn.get_occupied_orbs());  

        std::set_intersection(up_orbs.begin(), up_orbs.end(),
                            index_map.begin(), index_map.end(),
                            std::back_inserter(overlap_list_up));
        std::set_intersection(dn_orbs.begin(), dn_orbs.end(),
                            index_map.begin(), index_map.end(),
                            std::back_inserter(overlap_list_dn));

        //get the list of orbtials bellow the A_S and above A_S:
        auto n = up_orbs.size();

        // guard against bad indices
        if (AS_first > n || AS_last >= n) { "Error: A_S is not a sublist of A_L"; }

        // [0, AS_first)  — excludes AS_first
        std::vector<unsigned> left_up_orbs(
            up_orbs.begin(), up_orbs.begin() + AS_first);

        // (AS_last, end) — excludes AS_last
        std::vector<unsigned> right_up_orbs(
            (AS_last + 1 <= n) ? up_orbs.begin() + AS_last + 1 : up_orbs.end(),
            up_orbs.end());

        std::vector<unsigned> left_dn_orbs(
            dn_orbs.begin(), dn_orbs.begin() + AS_first);

        // (AS_last, end) — excludes AS_last
        std::vector<unsigned> right_dn_orbs(
            (AS_last + 1 <= n) ? dn_orbs.begin() + AS_last + 1 : dn_orbs.end(),
            dn_orbs.end());
        //test
            // print them
        //if (Verbose == true){
        //    print_orbs("Full determinant up", up_orbs);
        //    print_orbs("  left_up_orbs", left_up_orbs);
        //    print_orbs("  right_up_orbs", right_up_orbs);
        //    print_orbs("Full determinant dn", dn_orbs);
        //    print_orbs("  left_dn_orbs", left_dn_orbs);
        //    print_orbs("  right_dn_orbs", right_dn_orbs);
        //    print_orbs("reference orbitals: ", reference_occupied_orbitals);
        //}

        
        // number of electrons in A_S sublist
        ne_s_up = overlap_list_up.size();
        ne_s_dn = overlap_list_dn.size();

        Ne_S_up[j] = ne_s_up;
        Ne_S_dn[j] = ne_s_dn;

        int e_trans = abs(ne_s_up - n_up_s) + abs(ne_s_dn - n_dn_s); //number of electron exchanged with outside of A_S
        if (e_trans == 0){
            // need all the left occupations are 1, all the right occupations are 0
            if (left_up_orbs == reference_occupied_orbitals && left_dn_orbs == reference_occupied_orbitals
            && right_up_orbs.empty() && right_dn_orbs.empty()){
                count_Gs += 1; //Count of state without electron transfer
                Gs_index_map.push_back(j); //record the index of determinant belong to G_S
                Lambda += std::norm(wf_l.coefs[i_state_l][j]); //Lambda = \sum_j |cj|^2 for det in G_S
            }
            else {
                count_e_transter_outside +=1;
            }

        }
        else if (e_trans == 1) {
            count_1e_transfer += 1;
        }
        else if (e_trans == 2) {
            count_2e_transfer += 1;
        }
        else if (e_trans == 3) {
            count_3e_transfer += 1;
        }
        else if (e_trans == 4) {
            count_4e_transfer += 1;
        }
        else {
            count_higher_transfer += 1;
        }
    }
    // --------------------------
    // Report result (summary)
    // --------------------------
    if (Verbose == true){
    const int total_classified = count_Gs + count_1e_transfer + count_2e_transfer +
                                 count_3e_transfer + count_4e_transfer + count_higher_transfer;
    printf(" A_S target electrons: up=%d, dn=%d\n", n_up_s, n_dn_s);
    printf(" # determinants(in variational space):                %d\n", static_cast<int>(n_det));
    printf("   - 0e transfer AS <--> AL-AS:        \n");
    printf("            - 0e transfer within AL-AS (G_S):         %d\n", count_Gs);
    printf("            - >0e transfer within AL-AS:         %d\n", count_e_transter_outside);
    printf("   - 1e transfer AS <--> AL-AS :               %d\n", count_1e_transfer);
    printf("   - 2e transfer AS <--> AL-AS:               %d\n", count_2e_transfer);
    printf("   - 3e transfer AS <--> AL-AS:               %d\n", count_3e_transfer);
    printf("   - 4e transfer AS <--> AL-AS:               %d\n", count_4e_transfer);
    printf("   - >4e transfer AS <--> AL-AS:              %d\n", count_higher_transfer);
    printf("   - classified total:          %d\n", total_classified);
    print_orbs("index map of determinants in Gs: ", Gs_index_map);
    }
    
    return {Lambda, Gs_index_map};
}

//Compute state overlap:
    //a helper method to map the list of occupied orbital in A_S to list of occupeid orbital in A_L index
std::vector<size_t> map_list(
    const std::vector<unsigned int>& list_s, // the list of occupeid orbital of small active space
    const std::vector<size_t>& index_map // mapping list with index_map[index in A_S] = index in A_L
) {
    std::vector<size_t> list_s_mapped;
    list_s_mapped.reserve(list_s.size());

    for (size_t val : list_s) {
        if (val >= index_map.size()) {
            throw std::out_of_range("Value in list_s is out of index_map range.");
        }
        list_s_mapped.push_back(index_map[val]);
    }

    return list_s_mapped; //list of occupied orbital in A_L index 
}
//determine if the small active space occupation is a subset of large active space occupation:
bool is_subset(const std::vector<size_t>& up_orbs_s_mapped, const std::vector<unsigned>& up_orbs_l,
            const std::vector<size_t>& dn_orbs_s_mapped, const std::vector<unsigned>& dn_orbs_l) {
    for (int val : up_orbs_s_mapped) {
        if (std::find(up_orbs_l.begin(), up_orbs_l.end(), val) == up_orbs_l.end()) {
            return false; // Found one that's not in up_orbs_l
        }
    }
    for (int val : dn_orbs_s_mapped) {
        if (std::find(dn_orbs_l.begin(), dn_orbs_l.end(), val) == dn_orbs_l.end()) {
            return false; // Found one that's not in dn_orbs_l
        }
    }
    return true; // All found
}
std::complex<double> Compute_overlap(
    const Wavefunction& wf_l,//large active space wavefunction
    const Wavefunction& wf_s,//small active space wavefunction
    const std::vector<size_t>& index_map, //KS orbital index map of A_S in A_L
    const std::vector<size_t>& Gs_index_map,//index map of determinant in wf_l with projection on wf_s
    size_t i_state_l,//state of wf_l
    size_t i_state_s,//state of wf_s
    const double Lambda, //used as normalization factor
    const bool Renorm //renormalize the remanin det in GS? 
) {
    
    //Obtain the projected coefficient of wf_l to the determinant basis of wf_s:
    std::vector<std::complex<double>> wf_l_coef_proj(wf_s.dets.size()); // the projected wf_l to A_S active space
    //Normalize the projected coefficient
    double Norm = 0;
    for (size_t i_det_s=0; i_det_s < wf_s.dets.size(); ++i_det_s){ 
        const Det &det_s = wf_s.dets[i_det_s]; //determinants of small active space
        std::vector<unsigned> up_orbs_s(det_s.up.get_occupied_orbs()); 
        std::vector<unsigned> dn_orbs_s(det_s.dn.get_occupied_orbs());
        //mapped occupied orbital to A_L from A_S
        std::vector<size_t> up_orbs_s_mapped = map_list(up_orbs_s, index_map);
        std::vector<size_t> dn_orbs_s_mapped = map_list(dn_orbs_s, index_map);

        wf_l_coef_proj[i_det_s] = 0; //projected coefficient of i_det_s determinant

        for (size_t j_det_l : Gs_index_map) {//loop through the index of Gs determinant in A_L active space
            const Det &det_l = wf_l.dets[j_det_l]; //determinants of large active space
            std::vector<unsigned> up_orbs_l(det_l.up.get_occupied_orbs()); 
            std::vector<unsigned> dn_orbs_l(det_l.dn.get_occupied_orbs());  
            //does the current determinant have projection on i_det_s determinant of A_S?
            bool add_coef = is_subset(up_orbs_s_mapped, up_orbs_l,dn_orbs_s_mapped, dn_orbs_l);
            if (add_coef){
                wf_l_coef_proj[i_det_s] += wf_l.coefs[i_state_l][j_det_l];
            }

        }
        Norm += std::norm(wf_l_coef_proj[i_det_s]);
        //wf_l_coef_proj[i_det_s] = wf_l_coef_proj[i_det_s]/Lambda;  Lambda is not the normalization factor ! 
    }
    //Apply the normalization factor coefficient: 
    if (Renorm == true){
        for (size_t i_det_s=0; i_det_s < wf_s.dets.size(); ++i_det_s){
            wf_l_coef_proj[i_det_s] = wf_l_coef_proj[i_det_s]/Norm;
    }
    }


    //Compute state overlap : 
    std::complex<double> state_overlap = 0.0;
    for (size_t i_det_s=0; i_det_s < wf_s.dets.size(); ++i_det_s){
        state_overlap += std::conj(wf_l_coef_proj[i_det_s]) * wf_s.coefs[i_state_s][i_det_s]; // D* × C
        
    }

    return state_overlap;
}