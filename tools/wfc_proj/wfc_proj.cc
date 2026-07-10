#include <hps/src/hps.h>
#include <shci/src/det/det.h>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <algorithm>  // for std::find
#include <cstdio>
#include <shci/src/config.h> //read the wavefunction from configure file
#include "./wfc_help.h"

//This method try to compute the projected wfc to a smaller active space




int main(int argc, char *argv[]) {
  //get the configuration setup:
  //orbital list
  const auto& List_band_small = Config::get<std::vector<int>>("A_S");//list of band with small active space
  const auto& List_band_large = Config::get<std::vector<int>>("A_L");//list of band with large active space
  //number of electrons in small/large active space up and dn channel
  const int& n_up_s = Config::get<int>("n_up_s");
  const int& n_dn_s = Config::get<int>("n_dn_s");
  const int& n_up_l = Config::get<int>("n_up_l");
  const int& n_dn_l = Config::get<int>("n_dn_l");
  //wavefunction files
  const std::string& f_wfc_small = Config::get<std::string>("f_wfc_small");
  const std::string& f_wfc_large = Config::get<std::string>("f_wfc_large");
  //read the wavefunction
  std::ifstream serialized_wf_small(f_wfc_small, std::ios::binary); 
  Wavefunction wf_s = hps::from_stream<Wavefunction>(serialized_wf_small);
  std::ifstream serialized_wf_large(f_wfc_large, std::ios::binary); 
  Wavefunction wf_l = hps::from_stream<Wavefunction>(serialized_wf_large);

  std::cout << "The band list of small active space: ";
    for (int band : List_band_small) {
    std::cout << band << " ";
    }   
    std::cout << '\n';
  std::cout << "The band list of large active space: ";
    for (int band : List_band_large) {
    std::cout << band << " ";
    }   
    std::cout << '\n';
  std::cout << "Read small active space wfc from: " << f_wfc_small << '\n';
  std::cout << "Read large active space wfc from: " << f_wfc_large << '\n';

  const int&norbs_s = List_band_small.size(); // number of orbitals
  const int&norbs_l = List_band_large.size();

  std::vector<unsigned> orbs_small(norbs_s);//list of orbitals
  std::vector<unsigned> orbs_large(norbs_l);//list of orbitals
  
  // Print the list
  printf("Orbital list (small) re-indexed in FCIDUMP: ");
  for (unsigned i = 0; i < orbs_small.size(); ++i) {
      orbs_small[i] = i;
      printf("%u ", orbs_small[i]);
  }
  printf("\n");
  printf("Orbital list (large) re-indexed in FCIDUMP: ");
  for (unsigned i = 0; i < orbs_large.size(); ++i) {
      orbs_large[i] = i;
      printf("%u ", orbs_large[i]);
  }
  printf("\n");

  //map the A_S index to A_L:
  std::vector<std::size_t> index_map = map_indices(List_band_small, List_band_large);
  printf("The mapped index of orbitals in small active space --> large active space: \n");
  for (unsigned i = 0; i < index_map.size(); ++i){
    printf("%5d -->  %5d \n", i, index_map[i]);
  }
  printf("\n");

  //Loop through the state of small active space 
  int n_state_s = wf_s.coefs.size(); // number of states
  int n_det_s = wf_s.dets.size(); // number of slater determinants
  int n_state_l = wf_l.coefs.size(); // number of states
  int n_det_l = wf_l.dets.size(); // number of slater determinants
  printf("Large active space have n_state_l: %u , ", n_state_l);
  printf(" n_det_l: %d\n", n_det_l);
  printf("Small active space have n_state_s: %u , ", n_state_s);
  printf(" n_det_s: %d\n", n_det_s);   

  // Prepare storage for overlap norms
  std::vector<std::vector<double>> overlap_norms(n_state_s, std::vector<double>(n_state_l, 0.0));

  for (int i_state_s = 0; i_state_s < n_state_s; ++i_state_s) { //for each state n in small active space
    printf("===============State %u in small active space wfc(threshold 1e-3)===================\n", i_state_s);
    print_sorted_determinants(wf_s, i_state_s, 1e-3, true, orbs_small);
    for (int i_state_l = 0; i_state_l < n_state_l; ++i_state_l) {
      printf("-------projected State %u in large active space wfc----------------\n", i_state_l);

      //Compute lambda and find G_S
      LambdaResult Lambdaresult;  // declared once
      if (i_state_s ==0 && i_state_l ==0){
      Lambdaresult = Compute_Lambda(index_map, i_state_l, wf_l, n_up_s, n_dn_s,true); //the part of determinant without electron transfer to outside of A_S
      }
      else{
      Lambdaresult = Compute_Lambda(index_map, i_state_l, wf_l, n_up_s, n_dn_s,false);  
      }
      double Lambda = Lambdaresult.Lambda;
      if (i_state_s ==0){
        printf(" Lambda (sum of coefs over G_S): %+16.6e\n", static_cast<double>(Lambda));
      }
      std::vector<size_t> Gs_index_map = Lambdaresult.Gs_index_map;
      //test: Print determinant in Gs:
      print_sorted_determinants_in_Gs(wf_l, i_state_l, 1e-3, true, orbs_large, Gs_index_map);
      //Compute overlap with the i_state_l with the i_state_s
      std::complex<double> state_overlap = Compute_overlap(wf_l, wf_s, index_map, Gs_index_map, i_state_l, i_state_s, Lambda, false);
      printf("Overlap <%u(A_S)|%u(A_L)> : Re = %+16.3e, Im = %+16.3e\n",i_state_s, i_state_l, state_overlap.real(), state_overlap.imag());

      // Store squared norm (modulus squared)
      overlap_norms[i_state_s][i_state_l] = std::norm(state_overlap);
    }
  }

  // ===== Final Report =====
// ===== Final Report =====
printf("\n============== Final Report: Norms of Overlaps ==============\n");
for (int i_state_s = 0; i_state_s < n_state_s; ++i_state_s) {
    printf("Small state %u: ", i_state_s);

    // Find max norm and corresponding i_state_l
    double max_norm = -1.0;
    int max_index = -1;

    for (int i_state_l = 0; i_state_l < n_state_l; ++i_state_l) {
        double norm_val = overlap_norms[i_state_s][i_state_l];
        printf("%+16.3e ", norm_val);

        if (norm_val > max_norm) {
            max_norm = norm_val;
            max_index = i_state_l;
        }
    }

    printf("\n  |  Max @ large state %d with norm = %+16.3e", max_index, max_norm);
    printf("\n");
}


    return 0;
  };
    
