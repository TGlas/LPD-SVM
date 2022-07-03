
//
// CUDA-accelerated kernel computations between data points stored in
// sparse matrices, as well as functions for preparing the kernel space
// mapping, and for computing the compressed kernel matrix.
//


#include "definitions.h"
#include <algorithm>
#include <iostream>

using namespace std;


// Maximal number of memory blocks per variable groups in the GPU sparse matrix.
static constexpr uint blocks_per_group = (48 * 1024 - 8192) / 8;


//
// Multiply dense matrices: C := A * B
//  A: m times k
//  B: k times n
//  C: m times n
//  layout: char[3] for A, B, C, either "R" or "C" for row/column major
//
void cuda_matmul(uint device, uint m, uint k, uint n, DeviceArray<float> const& A, DeviceArray<float> const& B, DeviceArray<float>& C, std::string const& layout)
{
	// We simply forward the job to cuBLAS.

	assert(layout.size() == 3);
	assert(layout[0] == 'R' || layout[0] == 'r' || layout[0] == 'C' || layout[0] == 'c');
	assert(layout[1] == 'R' || layout[1] == 'r' || layout[1] == 'C' || layout[1] == 'c');
	assert(layout[2] == 'R' || layout[2] == 'r' || layout[2] == 'C' || layout[2] == 'c');
	assert(A.size() == m * k);
	assert(B.size() == k * n);
	assert(C.size() == m * n);
#define isRowMajor(x) ((x) == 'R' || (x) == 'r')
	float alpha = 1.0f, beta = 0.0f;

	if (isRowMajor(layout[2]))
	{
		// compute B^T * A^T = C^T
		if (cublasSgemm(
				cudaDevices[device].cublas_handle,
				isRowMajor(layout[1]) ? CUBLAS_OP_N : CUBLAS_OP_T,
				isRowMajor(layout[0]) ? CUBLAS_OP_N : CUBLAS_OP_T,
				n, m, k,
				&alpha,
				B.data(),
				isRowMajor(layout[1]) ? n : k,
				A.data(),
				isRowMajor(layout[0]) ? k : m,
				&beta,
				C.data(),
				n
			) != CUBLAS_STATUS_SUCCESS)
		{
			throwError("cublasSgemm (dense matrix multiplication) failed");
		}
	}
	else
	{
		// compute A * B = C
		if (cublasSgemm(
				cudaDevices[device].cublas_handle,
				isRowMajor(layout[0]) ? CUBLAS_OP_T : CUBLAS_OP_N,
				isRowMajor(layout[1]) ? CUBLAS_OP_T : CUBLAS_OP_N,
				m, n, k,
				&alpha,
				A.data(),
				isRowMajor(layout[0]) ? k : m,
				B.data(),
				isRowMajor(layout[1]) ? n : k,
				&beta,
				C.data(),
				m
			) != CUBLAS_STATUS_SUCCESS)
		{
			throwError("cublasSgemm (dense matrix multiplication) failed");
		}
	}
#undef isRowMajor
}

//
// Multiply dense matrices: C := A * B
//  A: m times k   (row major)
//  B: k times n   (column major)
//  C: m times n   (row major)
// Layouts are fixed, but strides are flexible.
//
void cuda_matmul(uint device, uint m, uint k, uint n, float const* A, float const* B, float* C, uint lda, uint ldc)
{
	// We simply forward the job to cuBLAS.
	float alpha = 1.0f, beta = 0.0f;

	// compute B^T * A^T = C^T
	if (cublasSgemm(
			cudaDevices[device].cublas_handle,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			n, m, k,
			&alpha,
			B,
			k,
			A,
			lda,
			&beta,
			C,
			ldc
		) != CUBLAS_STATUS_SUCCESS)
	{
		throwError("cublasSgemm (dense matrix multiplication) failed");
	}
}


//
// Compute the squared norms of rows of G in parallel.
//
__global__ void kernel_cuda_rows_norm2(uint n, uint b, float const* G, float* norm2)
{
	uint i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= n) return;

	float sum = 0.0f;
	for (uint j=0; j<b; j++)
	{
		float g = G[b*i + j];
		sum += g*g;
	}
	norm2[i] = sum;
}

//
// Return the squared norms of rows of G.
//
DeviceArray<float> cuda_rows_norm2(uint n, DeviceArray<float> const& G)
{
	DeviceArray<float> norm2(n, false);
	uint b = G.size() / n;
	uint p = (n + 1023) / 1024;
	kernel_cuda_rows_norm2<<<p, 1024>>>(n, b, G.data(), norm2.data());
	return std::move(norm2);
}


//
// Compute the inverse square root of D, truncating small values to
// zero. This kernel must run in a single thread block.
// input: D contains n numbers in ascending order.
// output: D contains n-b[0] leading zeros, followed by b[0]<=n numbers in descending order.
//         b[0] contains the number of valid coefficients (some of which might be zero).
//
__global__ void kernel_cuda_inv_sqrt_prepare_D(uint n, float* D, uint* b)
{
	__shared__ uint shared_nnz;

	if (threadIdx.x == 0)
	{
		// find cutoff point with binary search
		float cutoff = 1e-6f * D[n-1];
		int l = -1, u = n;
		while (u - l >= 2)
		{
			uint m = (l + u) / 2;
			if (D[m] <= cutoff) l = m;
			else u = m;
		}
		uint nnz = (n - u + 31) & (~31);
		if (nnz > n) nnz -= 32;
		b[0] = nnz;
		shared_nnz = nnz;
	}
	__syncthreads();

	// actual transformation
	uint nnz = shared_nnz;
	for (uint i=threadIdx.x; i<n; i+=blockDim.x)
	{
		if (i < n - nnz || D[i] <= 0.0f) D[i] = 0.0f;
		else D[i] = 1.0f / std::sqrt(D[i]);
	}
}

//
// In numpy terminology: M[:b] = D[n-b:] * M[n-b:]
//
__global__ void kernel_cuda_inv_sqrt_process_M(uint n, uint* b, float const* D, float* M)
{
	uint row = blockDim.x * blockIdx.x + threadIdx.x;
	if (row >= n) return;
	uint bb = *b;
	uint offset = n - bb;
	for (uint col=0; col<bb; col++) M[row + n*col] = D[offset + col] * M[row + n*(offset+col)];
}

//
// Compute an inverse matrix factor of M (an inverse square root).
//  [in]    M: n times n, column-major
//  [out]   M: n times b, column-major
// On entry, M is a positive definite symmetric matrix.
// On exit, it contains the inverse matrix square root in its first
// b columns. The remaining columns are invalid.
//
// The function is implemented in terms of eigen decomposition so it can
// deal with ill-conditioning. The resulting matrix is truncated to its
// first b columns, where b is the (numerical) rank of M, rounded up to
// the next multiple of 32. Small eigen values are truncated.
// The function returns b.
//
uint cuda_inv_factor(uint device, uint n, DeviceArray<float>& M)
{
	// eigen values
	DeviceArray<float> D(n, false);

	// allocate workspace
	int worksize = 0;
	if (cusolverDnSsyevd_bufferSize(
			cudaDevices[device].cusolver_handle,
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_LOWER,
			n,
			M.data(),
			n,
			D.data(),
			&worksize
		) != CUSOLVER_STATUS_SUCCESS)
	{
		throwError("cusolverDnDsyevd_bufferSize (eigen decomposition) failed");
	}
	DeviceArray<float> work(worksize, false);

	// actual eigen decomposition
	DeviceArray<int> info(1, true);
	auto status = cusolverDnSsyevd(
			cudaDevices[device].cusolver_handle,
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_LOWER,
			n,
			M.data(),
			n,
			D.data(),
			work.data(),
			worksize,
			info.data()
		);
	if (status != CUSOLVER_STATUS_SUCCESS)
	{
		int i = 0;
		info.to_cpu(&i);
		if (i == 0) throwError("cusolverDnDsyevd (eigen decomposition) failed with status " + std::to_string(status));
		else throwError("cusolverDnDsyevd (eigen decomposition) failed with status " + std::to_string(status) + " and error code " + std::to_string(i));
	}

	// inverse square root of the eigen values with truncation
	DeviceArray<uint> b(1, false);
	kernel_cuda_inv_sqrt_prepare_D<<<1, 1024>>>(n, D.data(), b.data());
	CheckKernelLaunch;

	// scale eigen vectors accordingly
	kernel_cuda_inv_sqrt_process_M<<<(n+31)/32, 32>>>(n, b.data(), D.data(), M.data());
	CheckKernelLaunch;

	// obtain the reduced dimension and resize M accordingly
	uint b_host = 0;
	b.to_cpu(&b_host);
	M.resize(n * b_host);

	return b_host;
}


//
// Gaussian kernel computation between sparse data matrices.
// This CUDA kernel shall be invoked with a 2D grid corresponding to the
// group structure of the data. It is designed to run with 1024 threads
// per block.
//
// The second data set shall be the basis. Memory access is optimized
// for the case that the point indices of the second data set are
// consecutive. Its size must be a multiple of 32.
//
__global__ void kernel_cuda_gauss \
		( \
			\                                        // first data set:
			uint rows1, \                            // number of rows
			uint const* offset1, \                   // data block offsets per variable group
			uint const* maxcol1, \                   // max column per data block
			GpuSparseMatrix::Block const* data1, \   // data blocks
			\
			\                                        // second data set:
			uint rows2, \                            // number of rows
			uint const* offset2, \                   // data block offsets per variable group
			uint const* maxcol2, \                   // max column per data block
			GpuSparseMatrix::Block const* data2, \   // data blocks
			\
			float kernel_gamma, \                    // kernel parameter
			\
			float* matrix \                          // dense kernel matrix (result)
		)
{
	// shared memory for non-zero elements reused many times
	__shared__ GpuSparseMatrix::Block block1;        // cached memory block of data set 1
	__shared__ GpuSparseMatrix::Block block2;        // cached memory block of data set 2
	__shared__ uint mc1[blocks_per_group];           // maximum columns for blocks of data set 1
	__shared__ uint mc2[blocks_per_group];           // maximum columns for blocks of data set 2

	// determine memory blocks to process
	uint o1 = offset1[2 * blockIdx.x];
	uint n1 = offset1[2 * blockIdx.x + 1];
	uint b1 = 0;
	uint o2 = offset2[2 * blockIdx.y];
	uint n2 = offset2[2 * blockIdx.y + 1];
	uint b2 = 0;
	__syncthreads();

	// load maximum column arrays
	for (uint i=threadIdx.x; i<n1; i += 1024)
	{
		if (i < n1) mc1[i] = maxcol1[o1 + i];
	}
	for (uint i=threadIdx.x; i<n2; i += 1024)
	{
		if (i < n2) mc2[i] = maxcol2[o2 + i];
	}
	__syncthreads();

	// load the first memory blocks of both data sets
	block1.raw[threadIdx.x] = data1[o1].raw[threadIdx.x];
	block2.raw[threadIdx.x] = data2[o2].raw[threadIdx.x];
	__syncthreads();

	// determine the role of this thread within the thread block
	uint i1 = threadIdx.x >> 5;   // index into block1
	uint i2 = threadIdx.x & 31;   // index into block2

	// compute the index into the result matrix
	uint row1 = block1.data.column[i1];   // major dimension of the result matrix
	uint row2 = block2.data.column[i2];   // minor dimension of the result matrix
	uint idx = row1 * rows2 + row2;       // index into the result matrix

	// determine index ranges for each thread
	uint s1 = block1.data.start_end[i1] & 0xffff;
	uint e1 = block1.data.start_end[i1] >> 16;
	uint s2 = block2.data.start_end[i2] & 0xffff;
	uint e2 = block2.data.start_end[i2] >> 16;

	// load the squared norms
	float sqnorms = block1.data.value[i1] + block2.data.value[i2];

	// inner product computed by this thread
	float accumulator = 0.0f;

	// block loop
	while (true)
	{
		// number crunching: loop through sparse matrix data and accumulate products
		if (s1 < e1 && s2 < e2)
		{
			uint c1 = block1.data.column[s1];
			uint c2 = block2.data.column[s2];
			while (true)
			{
				if (c1 == c2)
				{
					accumulator += block1.data.value[s1] * block2.data.value[s2];
					s1++;
					s2++;
					if (s1 >= e1 || s2 >= e2) break;
					c1 = block1.data.column[s1];
					c2 = block2.data.column[s2];
				}
				else if (c1 < c2)
				{
					s1++;
					if (s1 >= e1) break;
					c1 = block1.data.column[s1];
				}
				else
				{
					s2++;
					if (s2 >= e2) break;
					c2 = block2.data.column[s2];
				}
			}
		}
		__syncthreads();

		// check stopping criterion, load successor block(s)
		uint maxc1 = mc1[b1];
		uint maxc2 = mc2[b2];
		if (maxc1 <= maxc2)
		{
			b1++;
			if (b1 >= n1) break;
			block1.raw[threadIdx.x] = data1[o1 + b1].raw[threadIdx.x];
			__syncthreads();
			s1 = block1.data.start_end[i1] & 0xffff;
			e1 = block1.data.start_end[i1] >> 16;
		}
		if (maxc1 >= maxc2)
		{
			b2++;
			if (b2 >= n2) break;
			block2.raw[threadIdx.x] = data2[o2 + b2].raw[threadIdx.x];
			__syncthreads();
			s2 = block2.data.start_end[i2] & 0xffff;
			e2 = block2.data.start_end[i2] >> 16;
		}
	}
	__syncthreads();

	// compute the kernel value from the inner product
	accumulator *= 2.0f;
	accumulator -= sqnorms;
	accumulator *= kernel_gamma;
	accumulator = expf(accumulator);
	__syncthreads();

	// store the result
	if (row1 < rows1 && row2 < rows2) matrix[idx] = accumulator;
}

//
// Polynomial kernel computation between sparse data matrices.
// This CUDA kernel shall be invoked with a 2D grid corresponding to the
// block structure of the data. It is designed to run with as many
// threads per block as possible, since the kernel anyway uses the
// complete shared memory of the processor.
//
__global__ void kernel_cuda_poly \
		( \
			\                                        // first data set:
			uint rows1, \                            // number of rows
			uint const* offset1, \                   // data block offsets per variable group
			uint const* maxcol1, \                   // max column per data block
			GpuSparseMatrix::Block const* data1, \   // data blocks
			\
			\                                        // second data set:
			uint rows2, \                            // number of rows
			uint const* offset2, \                   // data block offsets per variable group
			uint const* maxcol2, \                   // max column per data block
			GpuSparseMatrix::Block const* data2, \   // data blocks
			\
			float kernel_gamma, \                    // kernel parameter
			float offset, \                          // kernel parameter
			float degree, \                          // kernel parameter
			\
			float* matrix \                          // dense kernel matrix (result)
		)
{
	// shared memory for non-zero elements reused many times
	__shared__ GpuSparseMatrix::Block block1;        // cached memory block of data set 1
	__shared__ GpuSparseMatrix::Block block2;        // cached memory block of data set 2
	__shared__ uint mc1[blocks_per_group];           // maximum columns for blocks of data set 1
	__shared__ uint mc2[blocks_per_group];           // maximum columns for blocks of data set 2

	// determine memory blocks to process
	uint o1 = offset1[2 * blockIdx.x];
	uint n1 = offset1[2 * blockIdx.x + 1];
	uint b1 = 0;
	uint o2 = offset2[2 * blockIdx.y];
	uint n2 = offset2[2 * blockIdx.y + 1];
	uint b2 = 0;
	__syncthreads();

	// load maximum column arrays
	for (uint i=threadIdx.x; i<n1; i += 1024)
	{
		if (i < n1) mc1[i] = maxcol1[o1 + i];
	}
	for (uint i=threadIdx.x; i<n2; i += 1024)
	{
		if (i < n2) mc2[i] = maxcol2[o2 + i];
	}
	__syncthreads();

	// load the first memory blocks of both data sets
	block1.raw[threadIdx.x] = data1[o1].raw[threadIdx.x];
	block2.raw[threadIdx.x] = data2[o2].raw[threadIdx.x];
	__syncthreads();

	// determine the role of this thread within the thread block
	uint i1 = threadIdx.x >> 5;   // index into block1
	uint i2 = threadIdx.x & 31;   // index into block2

	// compute the index into the result matrix
	uint row1 = block1.data.column[i1];   // major dimension of the result matrix
	uint row2 = block2.data.column[i2];   // minor dimension of the result matrix
	uint idx = row1 * rows2 + row2;       // index into the result matrix

	// determine index ranges for each thread
	uint s1 = block1.data.start_end[i1] & 0xffff;
	uint e1 = block1.data.start_end[i1] >> 16;
	uint s2 = block2.data.start_end[i2] & 0xffff;
	uint e2 = block2.data.start_end[i2] >> 16;

	// inner product computed by this thread
	float accumulator = 0.0f;

	// block loop
	while (true)
	{
		// number crunching: loop through sparse matrix data and accumulate products
		if (s1 < e1 && s2 < e2)
		{
			uint c1 = block1.data.column[s1];
			uint c2 = block2.data.column[s2];
			while (true)
			{
				if (c1 == c2)
				{
					accumulator += block1.data.value[s1] * block2.data.value[s2];
					s1++;
					s2++;
					if (s1 >= e1 || s2 >= e2) break;
					c1 = block1.data.column[s1];
					c2 = block2.data.column[s2];
				}
				else if (c1 < c2)
				{
					s1++;
					if (s1 >= e1) break;
					c1 = block1.data.column[s1];
				}
				else
				{
					s2++;
					if (s2 >= e2) break;
					c2 = block2.data.column[s2];
				}
			}
		}
		__syncthreads();

		// check stopping criterion, load successor block(s)
		uint maxc1 = mc1[b1];
		uint maxc2 = mc2[b2];
		if (maxc1 <= maxc2)
		{
			b1++;
			if (b1 >= n1) break;
			block1.raw[threadIdx.x] = data1[o1 + b1].raw[threadIdx.x];
			__syncthreads();
			s1 = block1.data.start_end[i1] & 0xffff;
			e1 = block1.data.start_end[i1] >> 16;
		}
		if (maxc1 >= maxc2)
		{
			b2++;
			if (b2 >= n2) break;
			block2.raw[threadIdx.x] = data2[o2 + b2].raw[threadIdx.x];
			__syncthreads();
			s2 = block2.data.start_end[i2] & 0xffff;
			e2 = block2.data.start_end[i2] >> 16;
		}
	}
	__syncthreads();

	// compute the kernel value from the inner product
	accumulator = powf(kernel_gamma * accumulator + offset, degree);
	__syncthreads();

	// store the result
	if (row1 < rows1 && row2 < rows2) matrix[idx] = accumulator;
}

//
// Sigmoid kernel computation between sparse data matrices.
// This CUDA kernel shall be invoked with a 2D grid corresponding to the
// block structure of the data. It is designed to run with as many
// threads per block as possible, since the kernel anyway uses the
// complete shared memory of the processor.
//
__global__ void kernel_cuda_tanh \
		( \
			\                                        // first data set:
			uint rows1, \                            // number of rows
			uint const* offset1, \                   // data block offsets per variable group
			uint const* maxcol1, \                   // max column per data block
			GpuSparseMatrix::Block const* data1, \   // data blocks
			\
			\                                        // second data set:
			uint rows2, \                            // number of rows
			uint const* offset2, \                   // data block offsets per variable group
			uint const* maxcol2, \                   // max column per data block
			GpuSparseMatrix::Block const* data2, \   // data blocks
			\
			float kernel_gamma, \                    // kernel parameter
			float offset, \                          // kernel parameter
			\
			float* matrix \                          // dense kernel matrix (result)
		)
{
	// shared memory for non-zero elements reused many times
	__shared__ GpuSparseMatrix::Block block1;        // cached memory block of data set 1
	__shared__ GpuSparseMatrix::Block block2;        // cached memory block of data set 2
	__shared__ uint mc1[blocks_per_group];           // maximum columns for blocks of data set 1
	__shared__ uint mc2[blocks_per_group];           // maximum columns for blocks of data set 2

	// determine memory blocks to process
	uint o1 = offset1[2 * blockIdx.x];
	uint n1 = offset1[2 * blockIdx.x + 1];
	uint b1 = 0;
	uint o2 = offset2[2 * blockIdx.y];
	uint n2 = offset2[2 * blockIdx.y + 1];
	uint b2 = 0;
	__syncthreads();

	// load maximum column arrays
	for (uint i=threadIdx.x; i<n1; i += 1024)
	{
		if (i < n1) mc1[i] = maxcol1[o1 + i];
	}
	for (uint i=threadIdx.x; i<n2; i += 1024)
	{
		if (i < n2) mc2[i] = maxcol2[o2 + i];
	}
	__syncthreads();

	// load the first memory blocks of both data sets
	block1.raw[threadIdx.x] = data1[o1].raw[threadIdx.x];
	block2.raw[threadIdx.x] = data2[o2].raw[threadIdx.x];
	__syncthreads();

	// determine the role of this thread within the thread block
	uint i1 = threadIdx.x >> 5;   // index into block1
	uint i2 = threadIdx.x & 31;   // index into block2

	// compute the index into the result matrix
	uint row1 = block1.data.column[i1];   // major dimension of the result matrix
	uint row2 = block2.data.column[i2];   // minor dimension of the result matrix
	uint idx = row1 * rows2 + row2;       // index into the result matrix

	// determine index ranges for each thread
	uint s1 = block1.data.start_end[i1] & 0xffff;
	uint e1 = block1.data.start_end[i1] >> 16;
	uint s2 = block2.data.start_end[i2] & 0xffff;
	uint e2 = block2.data.start_end[i2] >> 16;

	// inner product computed by this thread
	float accumulator = 0.0f;

	// block loop
	while (true)
	{
		// number crunching: loop through sparse matrix data and accumulate products
		if (s1 < e1 && s2 < e2)
		{
			uint c1 = block1.data.column[s1];
			uint c2 = block2.data.column[s2];
			while (true)
			{
				if (c1 == c2)
				{
					accumulator += block1.data.value[s1] * block2.data.value[s2];
					s1++;
					s2++;
					if (s1 >= e1 || s2 >= e2) break;
					c1 = block1.data.column[s1];
					c2 = block2.data.column[s2];
				}
				else if (c1 < c2)
				{
					s1++;
					if (s1 >= e1) break;
					c1 = block1.data.column[s1];
				}
				else
				{
					s2++;
					if (s2 >= e2) break;
					c2 = block2.data.column[s2];
				}
			}
		}
		__syncthreads();

		// check stopping criterion, load successor block(s)
		uint maxc1 = mc1[b1];
		uint maxc2 = mc2[b2];
		if (maxc1 <= maxc2)
		{
			b1++;
			if (b1 >= n1) break;
			block1.raw[threadIdx.x] = data1[o1 + b1].raw[threadIdx.x];
			__syncthreads();
			s1 = block1.data.start_end[i1] & 0xffff;
			e1 = block1.data.start_end[i1] >> 16;
		}
		if (maxc1 >= maxc2)
		{
			b2++;
			if (b2 >= n2) break;
			block2.raw[threadIdx.x] = data2[o2 + b2].raw[threadIdx.x];
			__syncthreads();
			s2 = block2.data.start_end[i2] & 0xffff;
			e2 = block2.data.start_end[i2] >> 16;
		}
	}
	__syncthreads();

	// compute the kernel value from the inner product
	accumulator = tanhf(kernel_gamma * accumulator + offset);
	__syncthreads();

	// store the result
	if (row1 < rows1 && row2 < rows2) matrix[idx] = accumulator;
}

//
// Kernel computation on the GPU.
//
DeviceArray<float> cuda_kernelmatrix(GpuSparseMatrix const& data1, GpuSparseMatrix const& data2, Kernel const& kernel)
{
	DeviceArray<float> result(data1.rows() * data2.rows(), false);

	// invoke the CUDA kernel
	dim3 grid(dim3(data1.numberOfGroups(), data2.numberOfGroups()));
	if (kernel.type == Kernel::Gaussian)
	{
		kernel_cuda_gauss<<<grid, 1024>>>
			(
				data1.rows(),
				data1.offset().data(),
				data1.maxcol().data(),
				data1.data().data(),

				data2.rows(),
				data2.offset().data(),
				data2.maxcol().data(),
				data2.data().data(),

				kernel.gamma,

				result.data()
			);
		CheckKernelLaunch;
	}
	else if (kernel.type == Kernel::Polynomial)
	{
		kernel_cuda_poly<<<grid, 1024>>>
			(
				data1.rows(),
				data1.offset().data(),
				data1.maxcol().data(),
				data1.data().data(),

				data2.rows(),
				data2.offset().data(),
				data2.maxcol().data(),
				data2.data().data(),

				kernel.gamma,
				kernel.offset,
				kernel.degree,

				result.data()
			);
		CheckKernelLaunch;
	}
	else if (kernel.type == Kernel::TanH)
	{
		kernel_cuda_tanh<<<grid, 1024>>>
			(
				data1.rows(),
				data1.offset().data(),
				data1.maxcol().data(),
				data1.data().data(),

				data2.rows(),
				data2.offset().data(),
				data2.maxcol().data(),
				data2.data().data(),

				kernel.gamma,
				kernel.offset,

				result.data()
			);
		CheckKernelLaunch;
	}
	else throwError("unknown kernel type");

	return std::move(result);
}
