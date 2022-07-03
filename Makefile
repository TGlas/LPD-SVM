
DEBUGFLAGS=-O2 -DNDEBUG
# DEBUGFLAGS=-g -DDEBUG

CPPFLAGS=-fopenmp -pthread -std=c++17 -fPIC -Wall -I/usr/local/cuda/include/ ${DEBUGFLAGS}
NVCCFLAGS=-Xcompiler -fopenmp -Xcompiler -pthread -std=c++14 -I/usr/local/cuda/include/ ${DEBUGFLAGS}
LINKERFLAGS=-fopenmp -pthread
NVCCLINKERFLAGS=-Xcompiler " -openmp" -Xcompiler " -pthread" -lgomp


all: cpu-svm gpu-svm convert-to-csr

cpu-svm: cpu-main.o aux.o cpu-svm.o eigen-solver.o
	g++ ${LINKERFLAGS} -o cpu-svm cpu-main.o aux.o cpu-svm.o eigen-solver.o

cpu-main.o: main.cpp definitions.h
	g++ ${CPPFLAGS} -o cpu-main.o -c main.cpp

cpu-svm.o: cpu-svm.cpp definitions.h
	g++ ${CPPFLAGS} -o cpu-svm.o -c cpu-svm.cpp

gpu-svm: gpu-main.o aux.o gpu-svm.o cpu-svm.o eigen-solver.o cuda-aux.o cuda-kernel.o cuda-smo.o
	nvcc ${NVCCLINKERFLAGS} -o gpu-svm -l cublas -l cusolver gpu-main.o aux.o gpu-svm.o cpu-svm.o eigen-solver.o cuda-aux.o cuda-kernel.o cuda-smo.o

gpu-main.o: main.cpp definitions.h
	g++ -DWITH_GPU_SUPPORT ${CPPFLAGS} -o gpu-main.o -c main.cpp

gpu-svm.o: gpu-svm.cpp definitions.h
	g++ -DWITH_GPU_SUPPORT ${CPPFLAGS} -o gpu-svm.o -c gpu-svm.cpp

aux.o: aux.cpp definitions.h
	g++ ${CPPFLAGS} -o aux.o -c aux.cpp

eigen-solver.o: eigen-solver.cpp definitions.h
	g++ ${CPPFLAGS} -o eigen-solver.o -c eigen-solver.cpp

cuda-aux.o: cuda-aux.cu definitions.h
	nvcc -DWITH_GPU_SUPPORT ${NVCCFLAGS} -o cuda-aux.o -c cuda-aux.cu

cuda-kernel.o: cuda-kernel.cu definitions.h
	nvcc -DWITH_GPU_SUPPORT ${NVCCFLAGS} -o cuda-kernel.o -c cuda-kernel.cu

cuda-smo.o: cuda-smo.cu definitions.h
	nvcc -DWITH_GPU_SUPPORT ${NVCCFLAGS} -o cuda-smo.o -c cuda-smo.cu

convert-to-csr: convert-to-csr.o
	g++ ${LINKERFLAGS} -o convert-to-csr convert-to-csr.o

convert-to-csr.o: convert-to-csr.cpp definitions.h
	g++ ${CPPFLAGS} -o convert-to-csr.o -c convert-to-csr.cpp

clean:
	rm -f *.o cpu-svm gpu-svm convert-to-csr
