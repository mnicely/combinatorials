NVCC	:=nvcc -lineinfo
CFLAGS	:=-O3 -std=c++14 -Xcompiler "-fopenmp"
ARCHES	:=-gencode arch=compute_70,code=\"compute_70,sm_70\" -gencode arch=compute_75,code=\"compute_75,sm_75\" -gencode arch=compute_80,code=\"compute_80,sm_80\" -gencode arch=compute_86,code=\"compute_86,sm_86\"
INC_DIR	:=
LIB_DIR	:=
LIBS	:=

SOURCES := combinatorials \
		   combosCheck

all: $(SOURCES)
.PHONY: all

combinatorials: combinatorials.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)
	
combosCheck: combosCheck.cu
	$(NVCC) $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)

clean:
	@echo 'Cleaning up...'
	@echo 'rm -rf $(SOURCES)'
	@rm -rf $(SOURCES) 
