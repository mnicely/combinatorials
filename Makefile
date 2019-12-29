NVCC	:=nvcc -lineinfo
CFLAGS	:=-O3 -std=c++14 -Xcompiler "-fopenmp"
ARCHES	:=-gencode arch=compute_75,code=\"compute_75,sm_75\" \
		-gencode arch=compute_70,code=\"compute_70,sm_70\" \
		-gencode arch=compute_60,code=\"compute_60,sm_60\"
INC_DIR	:=-I/usr/local/cuda/samples/common/inc
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
	rm -f $(SOURCES)
