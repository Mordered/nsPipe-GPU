NVCC = nvcc
LIBS = -lcufft -lcublas -lcurand -lcusparse
DEBUG = -g
GPU_SOURCES = $(wildcard src/*.cu)
GPU_OBJECTS = $(GPU_SOURCES:.cu=.o)

all: $(GPU_OBJECTS) 
	$(NVCC) -o Remesh $(GPU_OBJECTS) $(LIBS)

$(GPU_OBJECTS): src/%.o: src/%.cu 
	$(NVCC) -c   $< -o $@

clean:   
	rm src/*.o Remesh
