
//
// Interface definitions
//
// All interfaces are defined in terms of generic C++ types, without
// backend-specific (Eigen / CUDA) types (CUDA library handles are
// pointers). This allows for seamless combinations of optimized CPU
// and GPU code, despite NVCC's inability to compile the Eigen library.
//


#pragma once

#ifdef WITH_GPU_SUPPORT
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

#include <string>
#include <iostream>
#include <vector>
#include <list>
#include <unordered_map>
#include <tuple>
#include <utility>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <cstdint>
#include <cmath>
#include <cassert>

#include <cstdlib>


//
// 32 bit unsigned integer type, fits into one GPU register
//
using uint = std::uint32_t;

//
// exception with file name and line number information
//
#define throwError(what) throw std::runtime_error((std::string)__FILE__ + ":" + std::to_string(__LINE__) + " " + what)


#ifdef WITH_GPU_SUPPORT

//
// Per-GPU meta data and CUDA library handles
//
struct GpuData
{
	std::uint64_t global_memory_size;
	uint max_shared_memory_size;
	uint number_of_multiprocessors;
	bool kernel_timeout_enabled;
	bool initialized;
	cublasHandle_t cublas_handle;
	cusolverDnHandle_t cusolver_handle;
};
extern std::unordered_map<uint, GpuData> cudaDevices;


//
// Wrap each CUDA API call with this macro to check for errors.
//
void CheckCUDA_helper(cudaError_t code, char const* filename, int line);
#define CheckCUDA(code) { CheckCUDA_helper((code), __FILE__, __LINE__); }

//
// Include this macro after a kernel launch to check for errors.
// PERFORMANCE NOTE: forced synchronization in debug mode!
//
#ifdef DEBUG
#define CheckKernelLaunch \
CheckCUDA(cudaPeekAtLastError()); \
CheckCUDA(cudaDeviceSynchronize());
#else
#define CheckKernelLaunch \
CheckCUDA(cudaPeekAtLastError());
#endif

#endif


//
// Simple wall-clock-time timer.
//
class WallClockTimer
{
public:
	WallClockTimer()
	: m_start(std::chrono::high_resolution_clock::now())
	{ }

	void restart()
	{ m_start = std::chrono::high_resolution_clock::now(); }

	// obtain microseconds since start
	std::int64_t useconds() const
	{
		std::chrono::high_resolution_clock::time_point t = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::microseconds>(t - m_start).count();
	}

	// obtain seconds since start, at microseconds resolution
	double seconds() const
	{ return useconds() / 1e6; }

private:
	std::chrono::high_resolution_clock::time_point m_start;
};


//
// thread-safe console progress bar
//
class ProgressBar
{
public:
	static void launch();
	static void shutdown();

	static void start(std::size_t max, std::string const& description = "");
	static void increment(int delta = 1);
	static void stop();

private:
	static void threadmain();

	static std::atomic<bool> m_stop;
	static std::atomic<bool> m_active;
	static std::atomic<int> m_maximum;
	static std::atomic<int> m_value;
	static std::atomic<int> m_prevValue;
	static std::string m_description;
	static std::chrono::high_resolution_clock::time_point m_prevTime;
	static std::thread m_thread;
	static std::mutex m_mutex;
};

//
// Exponential range of the form base^low, ..., base^high.
// A single value is represented as base, with low=high=1.
//
struct Range
{
	Range()
	: base(0)
	, low(0)
	, high(0)
	{ }

	float base;
	int low, high;

	// access to values by index
	std::size_t size() const
	{ return high - low + 1; }
	float operator [] (std::size_t index) const
	{ return std::pow(base, (float)low + index); }

	// string representation of the range
	std::string toString() const;
};


//
// CPU memory representation of a dense matrix.
//
template <bool rowMajor>
struct DenseMatrix
{
	DenseMatrix()
	: rows(0)
	, cols(0)
	{ }

	DenseMatrix(std::size_t r, std::size_t c)
	: values(r * c)
	, rows(r)
	, cols(c)
	{ }

	void create(std::size_t r, std::size_t c)
	{
		values.resize(r * c);
		rows = r;
		cols = c;
	}

	DenseMatrix(DenseMatrix const& other) = default;
	DenseMatrix& operator = (DenseMatrix const& other) = default;
	DenseMatrix(DenseMatrix&& other) = default;
	DenseMatrix& operator = (DenseMatrix&& other) = default;

	float& operator () (std::size_t r, std::size_t c)
	{ return values[rowMajor ? r*cols+c : c*rows+r]; }
	float const& operator () (std::size_t r, std::size_t c) const
	{ return values[rowMajor ? r*cols+c : c*rows+r]; }

	float* data()
	{ return values.data(); }
	float const* data() const
	{ return values.data(); }

	std::vector<float> values;   // densely packed coefficients
	std::size_t        rows;     // number of rows
	std::size_t        cols;     // number of columns
};
using MatrixRM = DenseMatrix<true>;
using MatrixCM = DenseMatrix<false>;

//
// CPU memory representation of a sparse matrix in CSR format.
//
struct SparseMatrix
{
	SparseMatrix() : cols(0) { }
	SparseMatrix(SparseMatrix const& other) = default;
	SparseMatrix& operator = (SparseMatrix const& other) = default;
	SparseMatrix(SparseMatrix&& other) = default;
	SparseMatrix& operator = (SparseMatrix&& other) = default;

	std::vector<std::size_t> offset;   // begin/end of each row in the two arrays below
	std::vector<uint>        column;   // column index for each non-zero value
	std::vector<float>       value;    // non-zero value
	uint                     cols;     // maximal feature index + 1

	uint rows() const
	{ return offset.size() - 1; }
	std::size_t nnz() const
	{ return value.size(); }
	uint nnz(std::size_t row) const
	{ return offset[row+1] - offset[row]; }
};


#ifdef WITH_GPU_SUPPORT

//
// GPU library setup / cleanup wrappers
//
namespace gpu {
	void detectCudaDevices();
	uint selectMainCudaDevice(std::vector<uint> const& candidates);
	void init(uint device);
	void quit(uint device);
}

//
// RAII abstraction of CUDA device memory. The implementation of this
// class must be compiled with nvcc, but the declaration does not depend
// on CUDA.
//
class DeviceArrayBase
{
private:
	// (owning) device pointer
	void* m_data;

protected:
	// single-stage creation
	DeviceArrayBase(std::size_t bytes, bool zero);
	DeviceArrayBase(std::size_t bytes, void const* source);
	~DeviceArrayBase();

	// two-stage creation
	DeviceArrayBase()
	: m_data(nullptr)
	{ }
	void create(std::size_t bytes, bool zero);
	void create(std::size_t bytes, void const* source);
	void createDeviceCopy(size_t bytes, void const* source);   // very explicit copy of device memory

	// move, no copy
	DeviceArrayBase(DeviceArrayBase const& other) = delete;
	DeviceArrayBase& operator = (DeviceArrayBase const& other) = delete;
	DeviceArrayBase(DeviceArrayBase&& other);
	DeviceArrayBase& operator = (DeviceArrayBase&& other);

	// typed access to the device pointer
	template <typename T>
	T* ptr()
	{ return reinterpret_cast<T*>(m_data); }
	template <typename T>
	T const* ptr() const
	{ return reinterpret_cast<T const*>(m_data); }

	// copy operations between host and device
	bool to_gpu(void const* buffer, std::size_t start, std::size_t bytes);
	bool to_cpu(void* buffer, std::size_t start, std::size_t bytes) const;
};

//
// RAII abstraction of a typed array in CUDA device memory.
//
template <typename T>
class DeviceArray : DeviceArrayBase
{
private:
	std::size_t m_size;

public:
	// reserve memory on the device, initialize it to zero
	DeviceArray(std::size_t size, bool zero)
	: DeviceArrayBase(sizeof(T) * size, zero)
	, m_size(size)
	{ }

	// reserve memory on the device, initialize it by copying host memory
	DeviceArray(std::size_t size, T const* source)
	: DeviceArrayBase(sizeof(T) * size, source)
	, m_size(size)
	{ }
	DeviceArray(std::vector<T> const& source)
	: DeviceArrayBase(sizeof(T) * source.size(), source.data())
	, m_size(source.size())
	{ }

	// two-stage creation
	DeviceArray()
	: DeviceArrayBase()
	, m_size(0)
	{ }
	void create(std::size_t size, bool zero)
	{
		DeviceArrayBase::create(sizeof(T) * size, zero);
		m_size = size;
	}
	void create(std::size_t size, T const* source)
	{
		DeviceArrayBase::create(sizeof(T) * size, source);
		m_size = size;
	}
	void create(std::vector<T> const& source)
	{
		DeviceArrayBase::create(sizeof(T) * source.size(), source.data());
		m_size = source.size();
	}
	void createDeviceCopy(uint size, void const* source)   // very explicit copy of device memory
	{
		DeviceArrayBase::createDeviceCopy(sizeof(T) * size, source);
		m_size = size;
	}
	void createDeviceCopy(DeviceArray const& source)   // very explicit copy of device memory
	{
		DeviceArrayBase::createDeviceCopy(sizeof(T) * source.size(), source.data());
		m_size = source.size();
	}

	// move, no copy
	DeviceArray(DeviceArray const& other) = delete;
	DeviceArray& operator = (DeviceArray const& other) = delete;
	DeviceArray(DeviceArray&& other) = default;
	DeviceArray& operator = (DeviceArray&& other) = default;

	// access to size and data
	std::size_t size() const
	{ return m_size; }
	T* data()
	{ return ptr<T>(); }
	T const* data() const
	{ return ptr<T>(); }
	T* data(std::size_t offset)
	{ return ptr<T>() + offset; }
	T const* data(std::size_t offset) const
	{ return ptr<T>() + offset; }

	// copy operations between CPU and GPU (host and device, in CUDA terms)
	bool to_gpu(T const* buffer)
	{
		return DeviceArrayBase::to_gpu(buffer, 0, sizeof(T) * m_size);
	}
	bool to_gpu(T const* buffer, std::size_t start, std::size_t size)
	{
		assert(start + size <= m_size);
		return DeviceArrayBase::to_gpu(buffer, sizeof(T) * start, sizeof(T) * size);
	}
	bool to_gpu(std::vector<T> const& buffer)
	{
		assert(buffer.size() == m_size);
		return DeviceArrayBase::to_gpu(buffer.data(), 0, sizeof(T) * m_size);
	}

	bool to_cpu(T* buffer) const
	{
		return DeviceArrayBase::to_cpu(buffer, 0, sizeof(T) * m_size);
	}
	bool to_cpu(T* buffer, std::size_t start, std::size_t size) const
	{
		assert(start + size <= m_size);
		return DeviceArrayBase::to_cpu(buffer, sizeof(T) * start, sizeof(T) * size);
	}
	bool to_cpu(std::vector<T>& buffer) const
	{
		assert(buffer.size() == m_size);
		return DeviceArrayBase::to_cpu(buffer.data(), 0, sizeof(T) * m_size);
	}

	// Very limited resize. This function changes only the meta data of
	// the buffer, and hence it impacts the amount of memory transferred
	// between CPU and GPU. It does not release any memory.
	void resize(std::size_t newsize)
	{
		assert(newsize <= m_size);
		m_size = newsize;
	}
};


//
// GPU memory representation of a sparse matrix. The matrix data is
// packed into interleaved streams. Each stream holds the data of 32
// rows, or points, which are grouped together. The rows are stored in
// descending order of non-zeros, so that the number of non-zeros in
// each group is as homogeneous as possible. If necessary, the matrix
// is internally padded with zero rows so as to reach a multiple of 32.
// Each stream is split into blocks of 4KB. For each block, the class
// stores the maximal feature (column).
//
// For each stream or group of variables, the class holds the index of
// the starting block. The first 32 integers of each memory block hold
// start and end index of the actual data per variable, 16 bits each,
// packed into a single unsigned integer. The actual data follows: first
// up to 496 column indices, then up to 496 feature values. In the first
// block, the first 32 index/value pairs contain the "original" indices
// and the squared norms of the points. If the matrix is constructed
// from an index range then these indices still start at zero.
//
class GpuSparseMatrix
{
public:
	// 4KB memory block layout
	union Block
	{
		struct
		{
			uint start_end[32];
			uint column[496];
			float value[496];
		} data;
		uint raw[1024];
	};

	// Default constructor. Only use: assign later by move.
	GpuSparseMatrix()
	: m_rows(0)
	{ }

	// Prepare the matrix for fast kernel computation on the GPU.
	GpuSparseMatrix(SparseMatrix const& data);
	GpuSparseMatrix(SparseMatrix const& data, uint begin, uint end);

	// move, no copy
	GpuSparseMatrix(GpuSparseMatrix const& other) = delete;
	GpuSparseMatrix& operator = (GpuSparseMatrix const& other) = delete;
	GpuSparseMatrix(GpuSparseMatrix&& other) = default;
	GpuSparseMatrix& operator = (GpuSparseMatrix&& other) = default;

	// dimension (columns are inherently unknown)
	uint rows() const
	{ return m_rows; }
	uint numberOfGroups() const
	{ return m_offset.size() / 2; }

	// read access
	DeviceArray<uint> const& offset() const
	{ return m_offset; }
	DeviceArray<uint> const& maxcol() const
	{ return m_maxcol; }
	DeviceArray<Block> const& data() const
	{ return m_data; }

private:
	uint m_rows;
	DeviceArray<uint>  m_offset;   // starting block and number of blocks for each variable group
	DeviceArray<uint>  m_maxcol;   // maximal column for each block
	DeviceArray<Block> m_data;     // 4KB memory blocks, all in one big block
};

#endif


//
// Data structure describing the kernel function.
//
struct Kernel
{
	Kernel()
	: type(Gaussian)
	, gamma(0)
	, offset(0)
	, degree(3)
	{ }

	enum KernelType
	{
		Gaussian,
		Polynomial,
		TanH,
	};

	KernelType type;
	float gamma;
	float offset;
	float degree;

	// squared feature space norm of a data point
	float norm2(SparseMatrix const& data, uint index) const;

	// string representation of the kernel function and its parameters
	std::string toString(bool params = true) const;
};


//
// SVM model consisting of a list of class labels, a sparse matrix
// (data set) holding basis vectors, and a matrix of coefficients, with
// each binary problem corresponding to one column. When compiled with
// GPU support, a copy of the data in device memory is available,
// separately for each GPU.
//
struct Model
{
	Kernel             kernel;
	std::vector<float> classes;             // class labels present in the problem
	SparseMatrix       basis;               // basis functions / support vectors
	MatrixCM           alpha;               // one weight vector (column) per binary sub-problem
#ifdef WITH_GPU_SUPPORT
	std::unordered_map<uint, DeviceArray<float>> gpu_classes;   // same as classes, but in GPU memory, supporting multiple GPUs
	std::unordered_map<uint, GpuSparseMatrix>    gpu_basis;     // same as basis, but in GPU memory, supporting multiple GPUs
	std::unordered_map<uint, DeviceArray<float>> gpu_alpha;     // same as alpha, but in GPU memory, supporting multiple GPUs
#endif
	bool load(std::string const& filename);
	bool save(std::string const& filename) const;
};


//
// Thread pool with various job queues, designed to host one thread managing each GPU.
//
class WorkerPool
{
public:
	// create a dummy pool running all jobs in the main thread
	static void createSingle(uint id);

	// create a worker thread
	static void createPool(std::vector<uint> const& ids);

	// return the pool size, i.e., the number of concurrent threads
	static uint size()
	{ return (m_single >= 0) ? 1 : m_threads.size(); }

	static std::vector<uint> workerIDs()
	{ return m_ids; }

	// wait for the pool to complete all jobs (synchronization)
	static void wait();

	// stop all threads and shut down
	static void stop();

	// broadcast a job to all workers, to be processed with high priority
	static void broadcast(std::function<void(uint)> const& job);

	// broadcast a job to all workers; to be processed after all other jobs
	static void broadcastLast(std::function<void(uint)> const& job);

	// enqueue a job
	static void enqueue(std::function<void(uint)> const& job);

private:
	// worker thread main function
	static void main(uint id);

	static int m_single;                                                                  // single thread id, if >= 0
	static std::vector<uint> m_ids;                                                       // IDs of all workers
	static std::mutex m_mutex;                                                            // synchronization
	static std::condition_variable m_condition;                                           // trigger
	static std::atomic<uint> m_running;                                                   // number of running threads
	static std::atomic<uint> m_incomplete;                                                // number of pending jobs
	static std::atomic<bool> m_stop;                                                      // request to stop all threads
	static std::list<std::function<void(uint)>> m_jobs;                                   // pool-wide queue
	static std::unordered_map<uint, std::list<std::function<void(uint)>>> m_individual;   // thread-wise queue
	static std::unordered_map<uint, std::function<void(uint)>> m_individualLast;          // thread-wise trailing job
	static std::unordered_map<uint, std::thread> m_threads;                               // threads with GPU IDs
};


//
// general functionality
//
std::uint64_t availableRAM();
bool load(std::string const& filename, SparseMatrix& x, std::vector<float>& y, std::vector<float>& classes, std::vector<std::size_t>& permutation);
bool save(std::string const& filename, std::vector<float> const& predictions, std::vector<std::size_t> const& permutation);
float error(std::vector<float> const& labels, std::vector<float> const& predictions);
void print_approximation_error(SparseMatrix const& data, Kernel const& kernel, std::vector<float> const& norms);


//
// functions for training and prediction on the CPU
//
namespace cpu {
	std::tuple<uint, MatrixCM, MatrixRM, std::vector<float>> prepare(SparseMatrix const& x, SparseMatrix const& basis, Kernel const& kernel, uint budget);
	void train(MatrixCM const& V, MatrixRM const& G, std::vector<float> const& norm2, std::vector<float> const& labels, Model& model, float C, float epsilon);
	MatrixCM train_cv(MatrixRM const& G, std::vector<float> const& norm2, std::vector<float> const& labels, Model const& model, uint folds, Range const& C, float epsilon, bool warmstart);
	std::vector<float> predict(Model const& model, SparseMatrix const& data);
	std::vector<std::vector<float>> predict_cv(MatrixRM const& G, std::vector<float> const& classes, MatrixCM const& beta, uint folds);
};


//
// functions for training and prediction on the GPU
//
#ifdef WITH_GPU_SUPPORT
namespace gpu {
	std::tuple<uint, DeviceArray<float>, DeviceArray<float>, DeviceArray<float>> prepare_for_gpu(uint device, SparseMatrix const& x, GpuSparseMatrix const& basis, Kernel const& kernel, uint budget);
	std::tuple<uint, MatrixCM, MatrixRM, std::vector<float>> prepare_for_cpu(SparseMatrix const& x, std::unordered_map<uint, GpuSparseMatrix> const& basis, Kernel const& kernel, uint budget);
	void train(MatrixCM const& V, MatrixRM const& G, std::vector<float> const& norm2, std::vector<float> const& labels, Model& model, float C, float epsilon);
	void train(uint device, DeviceArray<float> const& V, DeviceArray<float> const& G, DeviceArray<float> const& norm2, std::vector<float> const& labels, Model& model, float C, float epsilon);
	MatrixCM train_cv(std::unordered_map<uint, DeviceArray<float>> const& G, std::vector<float> const& norm2, std::vector<float> const& labels, Model& model, uint folds, Range const& C, float epsilon, bool warmstart);
	std::vector<float> predict(Model const& model, SparseMatrix const& data);
	std::vector<std::vector<float>> predict_cv(std::unordered_map<uint, DeviceArray<float>> const& G, std::unordered_map<uint, DeviceArray<float>> const& classes, MatrixCM const& beta, uint folds);
	std::vector<std::vector<float>> predict_cv(MatrixRM const& G, std::unordered_map<uint, DeviceArray<float>> const& classes, MatrixCM const& beta, uint folds);
};

//
// GPU algorithms implemented in various *.cu files
//
void cuda_sync();
void cuda_matmul(uint device, uint m, uint k, uint n, DeviceArray<float> const& A, DeviceArray<float> const& B, DeviceArray<float>& C, std::string const& layout);
void cuda_matmul(uint device, uint m, uint k, uint n, float const* A, float const* B, float* C, uint lda, uint ldc);
DeviceArray<float> cuda_rows_norm2(uint n, DeviceArray<float> const& G);
uint cuda_inv_factor(uint device, uint n, DeviceArray<float>& M);
DeviceArray<float> cuda_kernelmatrix(GpuSparseMatrix const& data1, GpuSparseMatrix const& data2, Kernel const& kernel);
void cuda_prepare_binary(uint begin, uint end, uint classes, DeviceArray<uint> const& gpu_offset, DeviceArray<uint> const& gpu_yindex, DeviceArray<uint>& gpu_row, DeviceArray<float>& gpu_y);
void cuda_prepare_binary_cv(uint begin, uint end, uint Cs, uint folds, uint classes, DeviceArray<uint> const& gpu_offset, DeviceArray<uint> const& gpu_yindex, DeviceArray<uint>& gpu_row, DeviceArray<float>& gpu_y);
void cuda_smo(uint device, uint b, uint n, uint begin, uint end, std::vector<uint> const& sub_n, std::vector<uint> const& offset, DeviceArray<float> const& G, DeviceArray<float> const& norm2, DeviceArray<uint>& row, DeviceArray<float>& y, DeviceArray<float>& alpha, DeviceArray<float>& beta, std::vector<float> const& C, float epsilon, bool warmstart);
void cuda_vote(DeviceArray<float> const& votes, DeviceArray<float> const& classes, DeviceArray<float>& predictions);
#endif
