
//
// CUDA abstractions and helper functions.
//


#include "definitions.h"
#include <cooperative_groups.h>
#include <algorithm>
#include <iostream>
#include <map>

using namespace std;


// CUDA device properties
std::unordered_map<uint, GpuData> cudaDevices;


void CheckCUDA_helper(cudaError_t code, char const* filename, int line)
{
	if (code == cudaSuccess) return;
	throw std::runtime_error("CUDA error '" + std::string(cudaGetErrorString(code)) + "' in file '" + std::string(filename) + "', line " + std::to_string(line));
}


namespace gpu {

//
// Initialize the global list of GPUs and fill in their properties.
//
void detectCudaDevices()
{
	int count = 0;
	if (cudaGetDeviceCount(&count) == cudaSuccess)
	{
		for (int i=0; i<count; i++)
		{
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, i);
			if (properties.major != 9999)
			{
				cudaDevices[i].global_memory_size = properties.totalGlobalMem;
				cudaDevices[i].max_shared_memory_size = properties.sharedMemPerBlockOptin;
				cudaDevices[i].number_of_multiprocessors = properties.multiProcessorCount;
				cudaDevices[i].kernel_timeout_enabled = properties.kernelExecTimeoutEnabled;
				cudaDevices[i].initialized = false;
				cudaDevices[i].cublas_handle = nullptr;
				cudaDevices[i].cusolver_handle = nullptr;
			}
		}
	}
}

//
// Return the ID of the most powerful CUDA device.
//
uint selectMainCudaDevice(vector<uint> const& candidates)
{
	uint best = 0;
	uint64_t bestScore = 0;
	for (uint id : candidates)
	{
		uint64_t score = cudaDevices[id].number_of_multiprocessors * cudaDevices[id].global_memory_size;
		if (score > bestScore)
		{
			best = id;
			bestScore = score;
		}
	}
	return best;
}

//
// initialization and cleanup of CUDA related libraries
//
void init(uint device)
{
	if (! cudaDevices[device].initialized)
	{
		CheckCUDA(cudaSetDevice(device));

		if (cublasCreate(&cudaDevices[device].cublas_handle) != CUBLAS_STATUS_SUCCESS) throwError("failed to initialize cuBLAS");
		if (cusolverDnCreate(&cudaDevices[device].cusolver_handle) != CUSOLVER_STATUS_SUCCESS) throwError("failed to initialize cuSolver");

		cudaDevices[device].initialized = true;
	}
}

void quit(uint device)
{
	if (cudaDevices[device].initialized)
	{
		cublasDestroy(cudaDevices[device].cublas_handle);
		cusolverDnDestroy(cudaDevices[device].cusolver_handle);
		cudaDevices[device].cublas_handle = nullptr;
		cudaDevices[device].cusolver_handle = nullptr;
		cudaDevices[device].initialized = false;
	}
}

};


//
// Implementation of DeviceArrayBase.
// All CUDA API calls of the class are here.
//
DeviceArrayBase::DeviceArrayBase(size_t bytes, bool zero)
: m_data(nullptr)
{
	if (cudaMalloc(&m_data, bytes) != cudaSuccess) throwError("failed to allocate CUDA device memory");
	if (zero)
	{
		if (cudaMemset(m_data, 0, bytes) != cudaSuccess) throwError("failed to initialize CUDA device memory");
	}
}

DeviceArrayBase::DeviceArrayBase(size_t bytes, void const* source)
: m_data(nullptr)
{
	if (cudaMalloc(&m_data, bytes) != cudaSuccess) throwError("failed to allocate CUDA device memory");
	if (cudaMemcpy(m_data, source, bytes, cudaMemcpyHostToDevice) != cudaSuccess) throwError("failed to copy data to CUDA device memory");
}

DeviceArrayBase::~DeviceArrayBase()
{
	if (m_data) cudaFree(m_data);
}


void DeviceArrayBase::create(size_t bytes, bool zero)
{
	assert(! m_data);
	if (cudaMalloc(&m_data, bytes) != cudaSuccess) throwError("failed to allocate CUDA device memory");
	if (zero)
	{
		if (cudaMemset(m_data, 0, bytes) != cudaSuccess) throwError("failed to initialize CUDA device memory");
	}
}

void DeviceArrayBase::create(size_t bytes, void const* source)
{
	assert(! m_data);
	if (cudaMalloc(&m_data, bytes) != cudaSuccess) throwError("failed to allocate CUDA device memory");
	if (cudaMemcpy(m_data, source, bytes, cudaMemcpyHostToDevice) != cudaSuccess) throwError("failed to copy data to CUDA device memory");
}

void DeviceArrayBase::createDeviceCopy(size_t bytes, void const* source)
{
	assert(! m_data);
	if (cudaMalloc(&m_data, bytes) != cudaSuccess) throwError("failed to allocate CUDA device memory");
	if (cudaMemcpy(m_data, source, bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) throwError("failed to copy data within CUDA device memory");
}

DeviceArrayBase::DeviceArrayBase(DeviceArrayBase&& other)
{
	m_data = other.m_data;
	other.m_data = nullptr;
}

DeviceArrayBase& DeviceArrayBase::operator = (DeviceArrayBase&& other)
{
	if (m_data) cudaFree(m_data);

	m_data = other.m_data;
	other.m_data = nullptr;
	return *this;
}


bool DeviceArrayBase::to_gpu(void const* buffer, std::size_t start, size_t bytes)
{
	return (cudaMemcpy((char*)m_data + start, buffer, bytes, cudaMemcpyHostToDevice) == cudaSuccess);
}

bool DeviceArrayBase::to_cpu(void* buffer, std::size_t start, size_t bytes) const
{
	return (cudaMemcpy(buffer, (char const*)m_data + start, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
}


//
// warp-level reduction helpers for the kernel below
//
#if __CUDA_ARCH__ < 800
__device__ __forceinline__ uint warp_add_uint(uint value)
{
	value += __shfl_xor_sync(0xffffffff, value, 1);
	value += __shfl_xor_sync(0xffffffff, value, 2);
	value += __shfl_xor_sync(0xffffffff, value, 4);
	value += __shfl_xor_sync(0xffffffff, value, 8);
	value += __shfl_xor_sync(0xffffffff, value, 16);
	return value;
}
__device__ __forceinline__ uint warp_min_uint(uint value)
{
	value = fminf(value, __shfl_xor_sync(0xffffffff, value, 1));
	value = fminf(value, __shfl_xor_sync(0xffffffff, value, 2));
	value = fminf(value, __shfl_xor_sync(0xffffffff, value, 4));
	value = fminf(value, __shfl_xor_sync(0xffffffff, value, 8));
	value = fminf(value, __shfl_xor_sync(0xffffffff, value, 16));
	return value;
}
#else
__device__ __forceinline__ uint warp_add_uint(uint value)
{
	return __reduce_add_sync(0xffffffff, value);
}
__device__ __forceinline__ uint warp_min_uint(uint value)
{
	return __reduce_min_sync(0xffffffff, value);
}
#endif
__device__ __forceinline__ uint warp_partial_sum_uint(uint value)
{
	uint q1 = __shfl_xor_sync(0xffffffff, value, 1);
	uint sum2 = value + q1;
	uint q2 = __shfl_xor_sync(0xffffffff, sum2, 2);
	uint sum4 = sum2 + q2;
	uint q4 = __shfl_xor_sync(0xffffffff, sum4, 4);
	uint sum8 = sum4 + q4;
	uint q8 = __shfl_xor_sync(0xffffffff, sum8, 8);
	uint sum16 = sum8 + q8;
	uint q16 = __shfl_xor_sync(0xffffffff, sum16, 16);
	if (threadIdx.x & 1) value += q1;
	if (threadIdx.x & 2) value += q2;
	if (threadIdx.x & 4) value += q4;
	if (threadIdx.x & 8) value += q8;
	if (threadIdx.x & 16) value += q16;
	return value;
}

//
// CUDA kernel reorganizing a sparse matrix into memory blocks.
// Each block processes one group of 32 rows, and each thread is
// assigned to one row. The kernel must be called with a block size
// of 32.
//
__global__ void kernel_cuda_init_sparse_matrix(
		uint n, \                                 // number of data points (rows) in the matrix
		uint const* in_order, \                   // ordering of the rows
		uint const* in_offset, \                  // input sparse matrix row offsets
		uint const* in_column, \                  // input sparse matrix non-zero columns
		float const* in_value, \                  // input sparse matrix non-zero values
		uint* out_offset, \                       // output sparse matrix block offsets, pre-calculated start values
		uint* out_maxcol, \                       // output sparse matrix maximum column per block
		GpuSparseMatrix::Block* out_data) \       // output sparse matrix maximum block data
{
	bool valid = 32 * blockIdx.x + threadIdx.x < n;
	uint var = valid ? in_order[32 * blockIdx.x + threadIdx.x] : n;
	uint in = valid ? in_offset[var] : 0;         // index into in_column and in_value
	uint in_end = valid ? in_offset[var+1] : 0;   // end marker of the thread's row

	uint first = out_offset[2 * blockIdx.x];      // first memory block
	uint block = first;                           // current memory block

	float sqnorm = 0.0f;                          // squared norm accumulator
	uint s = 32;                                  // start index into block.data.column and block.data.value
	uint z = 32;                                  // number of entries used in the current block

	// store original index in the first block
	out_data[first].data.column[threadIdx.x] = var;

	// loop over blocks
	bool done = false;                            // stop if all threads have reached the end of their range
	while (! done)
	{
		// process one block

		// determine the maximal column index and the corresponding nnz for each thread
		uint in_break = in;
		uint col = (in_break < in_end) ? in_column[in_break] : 0xffffffff;
		uint current_col = warp_min_uint(col);
		uint max_col = 0xffffffff;
		while (current_col != 0xffffffff)
		{
			max_col = current_col;
			uint k = (col == current_col);
			in_break += k;
			z += warp_add_uint(k);
			if (z >= 496-31)
			{
				if (threadIdx.x == 0) out_maxcol[block] = current_col;
				break;
			}
			if (k) col = (in_break < in_end) ? in_column[in_break] : 0xffffffff;
			current_col = warp_min_uint(col);
		}
		done = (current_col == 0xffffffff);
		if (done && threadIdx.x == 0) out_maxcol[block] = max_col;

		// determine the index range within the memory block for each thread
		uint dest_end = in_break - in;
		dest_end = warp_partial_sum_uint(dest_end) + s;
		uint dest_start = dest_end - (in_break - in);
		out_data[block].data.start_end[threadIdx.x] = dest_start + (dest_end << 16);

		// copy column/value data into the memory block
		for ( ; in<in_break; in++)
		{
			uint col = in_column[in];
			float v = in_value[in];
			out_data[block].data.column[dest_start] = col;
			out_data[block].data.value[dest_start] = v;
			sqnorm += v * v;
			dest_start++;
		}

		// finalize the block and move on to the next one
		block++;
		s = 0;
		z = 0;
	}

	// store the squared norm in the first block
	out_data[first].data.value[threadIdx.x] = sqnorm;

	// store the number of used blocks
	if (threadIdx.x == 0) out_offset[2 * blockIdx.x + 1] = block - first;
}

//
// GpuSparseMatrix constructor, creating a GPU-suited copy of a CSR matrix.
//
GpuSparseMatrix::GpuSparseMatrix(SparseMatrix const& data)
: GpuSparseMatrix(data, 0, data.rows())
{ }

GpuSparseMatrix::GpuSparseMatrix(SparseMatrix const& data, uint begin, uint end)
: m_rows(end - begin)
{
	assert(end >= begin);
	assert(end <= data.rows());

	if (data.offset[end] - data.offset[begin] >= 0x0ffffffe0ULL) throwError("The data set contains too many non-zero features, exceeding the GPU implementation limit. Try processing the data on the CPU.");

	// Sort data by sparsity (decreasing order). This allows us to
	// process feature vectors of similar length together, minimizing
	// idle threads.
	vector<uint> order(m_rows);
	for (uint i=0; i<m_rows; i++) order[i] = i;
	std::sort(order.begin(), order.end(), [&data, begin] (uint a, uint b) { return (data.nnz(begin + a) > data.nnz(begin + b)); });

	// determine the number of data blocks
	uint groups = (end - begin + 31) / 32;
	vector<uint> offset(2 * groups);
	uint b = 0;
	for (uint g=0; g<groups; g++)
	{
		offset[2 * g] = b;
		uint nnz = 0;
		for (uint i=0; i<32; i++)
		{
			if (begin + 32 * g + i >= end) break;
			nnz += data.nnz(begin + order[32 * g + i]);
		}
		b += (nnz + 32 + 496 - 30) / (496 - 31);   // minimally wasteful, but parallelizable
	}

	// move the CSR matrix to GPU memory
	size_t ofs = data.offset[begin];
	size_t nnz = data.offset[end] - ofs;
	vector<uint> data_offset(end - begin + 1);
	for (uint i=begin; i<=end; i++) data_offset[i-begin] = data.offset[i] - ofs;
	DeviceArray<uint> in_offset(data_offset);
	DeviceArray<uint> in_column(nnz, &data.column[ofs]);
	DeviceArray<float> in_value(nnz, &data.value[ofs]);
	DeviceArray<uint> in_order(order);

	// allocate GPU memory for the result
	m_offset.create(offset);
	m_maxcol.create(b, false);
	m_data.create(b, false);

	// fill the memory blocks on the GPU
	kernel_cuda_init_sparse_matrix<<<groups, 32>>>(
			end - begin,
			in_order.data(),
			in_offset.data(),
			in_column.data(),
			in_value.data(),
			m_offset.data(),
			m_maxcol.data(),
			m_data.data()
		);
	CheckKernelLaunch;
}


//
// The kernel builds histograms of votes in shared memory and then
// determines the pointwise arg-max. It returns the corresponding class
// labels.
//
__global__ void kernel_cuda_vote(uint n, uint c, uint p, float const* votes, float const* classes, float* predictions)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) return;

	extern __shared__ uint sharedmem[];

	// prepare the histogram
	uint* histogram = sharedmem + c * threadIdx.x;
	for (uint k=0; k<c; k++) histogram[k] = 0;

	// collect votes
	uint a=0, b=1;
	for (uint k=0; k<p; k++)
	{
		if (votes[p * index + k] > 0.0f) histogram[b]++; else histogram[a]++;
		b++; if (b == c) { a++; b = a+1; }
	}

	// find the maximum
	uint best_i = 0, best_n = histogram[0];
	for (uint v=1; v<c; v++)
	{
		if (histogram[v] > best_n) { best_i = v; best_n = histogram[v]; }
	}
	predictions[index] = classes[best_i];
}

//
// Turn binary predictions ("votes") into multi-class predictions.
// The votes need to be organized in an n x p row-major matrix, where
// n is the number of data points and p is the number of binary
// sub-problems.
//
void cuda_vote(DeviceArray<float> const& votes, DeviceArray<float> const& classes, DeviceArray<float>& predictions)
{
	uint n = predictions.size();
	uint c = classes.size();
	uint p = c * (c - 1) / 2;
	assert(votes.size() == n * p);

	uint blocksize = 1024;
	while (blocksize * c > 12288) blocksize /= 2;
	if (blocksize == 0) blocksize = 1;

	kernel_cuda_vote<<<(n+blocksize-1)/blocksize, blocksize, 4*blocksize*c>>>(n, c, p, votes.data(), classes.data(), predictions.data());
	CheckKernelLaunch;
}


//
// explicit device synchronization, useful mostly for time measurements
//
void cuda_sync()
{
	CheckCUDA(cudaDeviceSynchronize());
}
