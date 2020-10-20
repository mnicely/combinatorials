NVCC	:=nvcc -lineinfo
CFLAGS	:=-O3 -std=c++14 -Xcompiler "-fopenmp"
ARCHES	:=-gencode arch=compute_75,code=\"compute_75,sm_75\" 
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
	rm -f $(SOURCES)
