SHCI_DIR := ../..

LIB_DIR := $(SHCI_DIR)/lib
PARENT_DIR := $(SHCI_DIR)/..
SRC := wf_rotate_back.cc
UTIL := $(SHCI_DIR)/src/util.cc
HDET := $(SHCI_DIR)/src/det/half_det.cc

all : $(SRC)
	g++ -std=c++17 -O2 -I$(LIB_DIR) -I$(PARENT_DIR) $(UTIL) $(HDET) $(SRC) -fopenmp -o wf_rotate_back

clean :
	$(RM) wf_rotate_back
