# Just change SHCI_DIR to the location of the SHCI directory
SHCI_DIR := ../..
OBJ_DIR := $(SHCI_DIR)/build_cray #added 

LIB_DIR := $(SHCI_DIR)/lib
PARENT_DIR := $(SHCI_DIR)/..
SRC := wfc_proj.cc
UTIL := $(SHCI_DIR)/src/util.cc
HDET := $(SHCI_DIR)/src/det/half_det.cc

all : wfc_proj.cc
	g++ -I$(LIB_DIR) -I$(PARENT_DIR) $(UTIL) $(HDET) $(SRC) -fopenmp -o wfc_proj
clean :
	$(RM) wfc_proj
