
//
// CUDA implementation of SMO-based SVM training.
//


#include "definitions.h"

using namespace std;


//
// Prepare one binary problem per block. The kernel is run with
// 32 threads per block. It loops through the data set and places
// points and labels into "row" and "y".
//
__global__ void kernel_cuda_prepare_binary(uint n, uint begin, uint classes, uint const* offset, uint const* yindex, uint* row, float* y)
{
	// sub-problem index
	uint k = begin + blockIdx.x;
	uint ofs = offset[k] - offset[begin];

	// class indices of this sub-problem
	uint neg = (uint)(classes - sqrt(classes * (classes-1) - 2*k + 0.25f) - 0.5f);
	uint pos = k - neg * (2*classes - neg - 3) / 2 + 1;

	// loop over the dataset
	uint m = (n + 31) & ~31;
	for (uint i=threadIdx.x; i<m; i+=32)
	{
		__syncthreads();
		uint c = (i < n) ? yindex[i] : 0xffffffff;
		bool store = (c == neg) || (c == pos);

		// find the index where to store the result
		uint active = __ballot_sync(0xffffffff, store);
		uint below = active & ((1u << threadIdx.x) - 1);

		// popcount method from here: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
		uint index = below - ((below >> 1) & 0x55555555);
		index = (index & 0x33333333) + ((index >> 2) & 0x33333333);
		index = ((index + (index >> 4) & 0xf0f0f0f) * 0x1010101) >> 24;

		// store the point
		if (store)
		{
			row[ofs + index] = i;                          // point/row
			y[ofs + index] = (c == neg) ? -1.0f : +1.0f;   // label
		}

		// update the offset
		__syncthreads();
		ofs += __shfl_sync(0xffffffff, index, 31);
		ofs += (active >> 31);
	}
}


//
// Prepare one binary problem per block. The kernel is run with
// 32 threads per block. It loops through the data set and places
// points and labels into "row" and "y".
//
__global__ void kernel_cuda_prepare_binary_cv(uint n, uint begin, uint Cs, uint folds, uint classes, uint const* offset, uint const* yindex, uint* row, float* y)
{
	// sub-problem index
	uint k = begin + blockIdx.x;
	uint ofs = offset[k] - offset[begin];
	uint p = (classes-1) * classes / 2;
	uint fold = (k / p) % folds;
	k = k % p;

	// class indices of this sub-problem
	uint neg = (uint)(classes - sqrt(classes * (classes-1) - 2*k + 0.25f) - 0.5f);
	uint pos = k - neg * (2*classes - neg - 3) / 2 + 1;

	// loop over the dataset
	uint m = (n + 31) & ~31;
	for (uint i=threadIdx.x; i<m; i+=32)
	{
		__syncthreads();
		uint c = (i < n) ? yindex[i] : 0xffffffff;
		uint f = i % folds;
		bool store = ((c == neg) || (c == pos)) && (f != fold);

		// find the index where to store the result
		uint active = __ballot_sync(0xffffffff, store);
		uint below = active & ((1u << threadIdx.x) - 1);

		// popcount method from here: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
		uint index = below - ((below >> 1) & 0x55555555);
		index = (index & 0x33333333) + ((index >> 2) & 0x33333333);
		index = ((index + (index >> 4) & 0xf0f0f0f) * 0x1010101) >> 24;

		// store the point
		if (store)
		{
			row[ofs + index] = i;                          // point/row
			y[ofs + index] = (c == neg) ? -1.0f : +1.0f;   // label
		}

		// update the offset
		__syncthreads();
		ofs += __shfl_sync(0xffffffff, index, 31);
		ofs += (active >> 31);
	}
}


//
// Prepare binary problem descriptions in the range begin:end.
// This amounts to filling in gpu_row and gpu_y with points (rows) and labels.
// The parameters have the following meaning:
//   begin         start of the range
//   end           end of the range
//   classes       number of classes
//   offset        sub-problem start index, i.e., sub of all previous sub-problems
//   gpu_yindex    class indices of all data points
//
void cuda_prepare_binary(uint begin, uint end, uint classes, DeviceArray<uint> const& gpu_offset, DeviceArray<uint> const& gpu_yindex, DeviceArray<uint>& gpu_row, DeviceArray<float>& gpu_y)
{
	kernel_cuda_prepare_binary<<<end - begin, 32>>>(gpu_yindex.size(), begin, classes, gpu_offset.data(), gpu_yindex.data(), gpu_row.data(), gpu_y.data());
	CheckKernelLaunch;
}

//
// Prepare binary cross-validation problem descriptions in the range begin:end.
// This amounts to filling in gpu_row and gpu_y with points (rows) and labels.
// The parameters have the following meaning:
//   begin         start of the range
//   end           end of the range
//   Cs            number of C-values
//   folds         number of folds
//   classes       number of classes
//   offset        sub-problem start index, i.e., sub of all previous sub-problems
//   gpu_yindex    class indices of all data points
//
void cuda_prepare_binary_cv(uint begin, uint end, uint Cs, uint folds, uint classes, DeviceArray<uint> const& gpu_offset, DeviceArray<uint> const& gpu_yindex, DeviceArray<uint>& gpu_row, DeviceArray<float>& gpu_y)
{
	kernel_cuda_prepare_binary_cv<<<end - begin, 32>>>(gpu_yindex.size(), begin, Cs, folds, classes, gpu_offset.data(), gpu_yindex.data(), gpu_row.data(), gpu_y.data());
	CheckKernelLaunch;
}


//
// GPU sorting, needed to bring alpha back into the original order after training with shrinking.
//

// crossover comparison with the given block size
__global__ void kernel_cuda_bitonic_crossover(uint offset0, uint const* a_offset, uint const* a_size, uint* key, float* data, uint blocksize)
{
	// problem properties
	uint start = a_offset[blockIdx.y] - offset0;
	uint n = a_size[blockIdx.y];
	key += start;
	data += start;

	// addresses
	uint thread = blockIdx.x * blockDim.x + threadIdx.x;
	uint offset = blocksize / 2;
	uint low = thread & (offset - 1);
	uint i0 = low | (2 * (thread & ~(offset - 1)));
	uint i1 = i0 + blocksize - 1 - 2 * low;
	if (i1 >= n) return;

	// perform a single crossover comparison+swap
	uint k0 = key[i0];
	uint k1 = key[i1];
	if (k0 > k1)
	{
		float d0 = data[i0];
		float d1 = data[i1];
		key[i0] = k1;
		key[i1] = k0;
		data[i0] = d1;
		data[i1] = d0;
	}
}

// linear (non-crossover) comparison with the given block size over a range up to 10 block sizes
__global__ void kernel_cuda_bitonic_linear_block(uint offset0, uint const* a_offset, uint const* a_size, uint* key, float* data, uint maxblocksize, uint minblocksize)
{
	// 16K shared memory
	__shared__ uint k[2048];
	__shared__ float d[2048];

	// problem properties
	uint start = a_offset[blockIdx.y] - offset0;
	uint n = a_size[blockIdx.y];
	key += start;
	data += start;

	// addresses
	uint thread = threadIdx.x;
	uint block = blockIdx.x;
	uint stride = minblocksize / 2;
	uint low = block & (stride - 1);
	uint high = block / stride;
	uint index = low + stride * (thread + 2048 * high);
	uint offset = 1024 * stride;

	// load data
	if (index < n)
	{
		k[thread] = key[index];
		d[thread] = data[index];
	}
	else k[thread] = INFINITY;
	__syncthreads();
	if (index + offset < n)
	{
		k[thread + 1024] = key[index + offset];
		d[thread + 1024] = data[index + offset];
	}
	else k[thread + 1024] = INFINITY;
	__syncthreads();

	// perform sweeps of linear comparisons
	for (uint ofs = maxblocksize / minblocksize; ofs >= 1; ofs /= 2)
	{
		uint low = threadIdx.x & (ofs - 1);
		uint high = threadIdx.x & ~(ofs - 1);
		uint i0 = low | (2 * high);
		uint i1 = i0 + ofs;

		uint k0 = k[i0];
		uint k1 = k[i1];
		if (k0 > k1)
		{
			float d0 = d[i0];
			float d1 = d[i1];
			k[i0] = k1;
			k[i1] = k0;
			d[i0] = d1;
			d[i1] = d0;
		}

		__syncthreads();
	}

	// store data
	if (index < n)
	{
		key[index] = k[thread];
		data[index] = d[thread];
	}
	__syncthreads();
	if (index + offset < n)
	{
		key[index + offset] = k[thread + 1024];
		data[index + offset] = d[thread + 1024];
	}
}

//
// Sort a range of arrays in-place. The arrays are sub-arrays of "key"
// and "data", starting at offset[i]-offset0 and taking up n_sub[i]
// elements each.
//
// The function sorts "key" while keeping the order of "data" consistent
// with "key". "n_max" is the maximum over n_sub, i.e., the largest
// sub-array size.
//
void cuda_sort(uint offset0, DeviceArray<uint> const& offset, DeviceArray<uint> const& n_sub, uint n_max, DeviceArray<uint>& key, DeviceArray<float>& data)
{
	uint m = (n_max + 1023) / 1024;
	dim3 gridsize(m, offset.size());
	for (uint k=2; k<2*n_max; k *= 2)
	{
		kernel_cuda_bitonic_crossover<<<gridsize, 1024>>>(offset0, offset.data(), n_sub.data(), key.data(), data.data(), k);
		CheckKernelLaunch;

		uint j = k/2;
		while (j >= 2)
		{
			uint jj = std::max<uint>(2, j / 512);
			kernel_cuda_bitonic_linear_block<<<gridsize, 1024>>>(offset0, offset.data(), n_sub.data(), key.data(), data.data(), j, jj);
			CheckKernelLaunch;
			j = jj / 2;
		}
	}
}


//
// Data describing a binary classification problem and its status.
//
struct Problem
{
	Problem() = default;
	Problem(Problem const&) = default;
	Problem& operator = (Problem const&) = default;

	uint alpha_offset;       // offset into variable arrays
	uint beta_offset;        // offset into compressed variable array
	uint n;                  // number of variables
	uint active;             // number of active variables
	float C;                 // complexity control parameter
	float epsilon;           // target precision
	float violation;         // current maximal KKT violation
	uint epoch;              // completed number of epochs
	uint iterations;         // number of SMO iterations (may overflow)
	uint unshrink_counter;   // iterations without unshrinking
	uint solved;             // 0/1
};


//
// SMO solver CUDA kernel, running as a single thread block per problem
// instance since it needs to access the same block of shared memory.
// The kernel can run as a grid, where each instance solves an
// independent SVM training problem sharing the same data. This is
// useful for cross-validation, training with multiple values of C
// simultaneously (e.g., parameter tuning), and for multi-class
// problems.
//
// The feature dimension b is required to be a multiple of 32, and 64 at
// least. The kernel must be invoked with z = min(1024, b) threads. The
// number of threads (block size) is also used as a batch size (z). The
// kernel must be invoked with a shared memory size of at least
//      40*z + 8*b + 256 bytes.
// With 49152 bytes of shared memory, this limits us to b <= 992. For
// modern server GPUs, the full shared memory size is used to allow for
// larger budgets. There is an implementation limit of b <= 16384. As of
// now (12/2021), this limit suffices for all available GPU models.
//
// The kernel is a template, with the dimension and the batch size as
// parameters. This enables loop unrolling.
//
// All parameters with prefix "a_" are arrays with separate entries per
// problem instance.
//
// Depending on the setup, CUDA kernels are subject to (OS dependent)
// runtime limits. Therefore, this kernel executes a number of epochs
// until a maximal number of iterations has completed. This means that
// the kernel needs to be invoked multiple times until the precision
// meets the target. In between calls, solved problems should be
// removed. To avoid unnecessary communication overheads and at the same
// time be on the safe side, the iteration limit should be adapted such
// that the kernel runs for roughly one second.
//
template <uint b, uint z>
__global__ void kernel_cuda_smo( \
		int maxiter, \                     // target number of SMO iterations
		float const* G, \                  // row-wise compressed kernel matrix
		float const* G_norm2, \            // squared norms of the rows of G
		uint* a_row, \                     // problem specific rows of G
		float* a_label, \                  // problem specific labels (+1/-1)
		uint* a_shrink, \                  // problem specific shrink counters
		float* a_alpha, \                  // problem specific uncompressed weights
		float* a_beta, \                   // problem specific compressed weights
		Problem* a_problem)                // problem specific data, including status
{
	// dynamically sized shared memory
	extern __shared__ float sharedmem[];

// macros for accessing shared memory
#define batchsize      z
#define baseindex      ((index < high) ? 0 : batchsize)
#define alpha          sharedmem
#define alphaH         (sharedmem + batchsize)
#define row            (sharedmem + 2*batchsize)
#define rowH           (sharedmem + 3*batchsize)
#define norm2          (sharedmem + 4*batchsize)
#define norm2H         (sharedmem + 5*batchsize)
#define label          (sharedmem + 6*batchsize)
#define labelH         (sharedmem + 7*batchsize)
#define shrink         ((uint*)(void*)(sharedmem + 8*batchsize))
#define shrinkH        ((uint*)(void*)(sharedmem + 9*batchsize))
#define beta           (sharedmem + 10*batchsize)
#define G_row          (sharedmem + 10*batchsize + b)
#define intermediate   (sharedmem + 10*batchsize + 2*b)
#define shared_step    sharedmem[10*batchsize + 2*b + 32]
#define shared_vio     sharedmem[10*batchsize + 2*b + 33]
#define SWAP(a, b) { auto temp = a; a = b; b = temp; }
#define shrink_threshold   5

	// problem specific array offsets
	{
		const uint alpha_offset = a_problem[blockIdx.x].alpha_offset;
		const uint beta_offset = a_problem[blockIdx.x].beta_offset;
		a_alpha += alpha_offset;
		a_row += alpha_offset;
		a_label += alpha_offset;
		a_shrink += alpha_offset;
		a_beta += beta_offset;
	}

	const uint n = a_problem[blockIdx.x].n;
	uint active = a_problem[blockIdx.x].active;
	const float C = a_problem[blockIdx.x].C;
	const float epsilon = a_problem[blockIdx.x].epsilon;
	uint epoch = a_problem[blockIdx.x].epoch;
	uint unshrink_counter = a_problem[blockIdx.x].unshrink_counter;

	uint index = 0;
	uint high = ((active-1) / batchsize) * batchsize;

	// load "high" buffer segments
	if (high + threadIdx.x < n)
	{
		alphaH[threadIdx.x] = a_alpha[high + threadIdx.x];
		uint r = a_row[high + threadIdx.x];
		rowH[threadIdx.x] = r;
		norm2H[threadIdx.x] = G_norm2[r];
		labelH[threadIdx.x] = a_label[high + threadIdx.x];
		shrinkH[threadIdx.x] = a_shrink[high + threadIdx.x];
	}

	// load beta
	#pragma unroll
	for (uint i=threadIdx.x; i<b; i+=z) beta[i] = a_beta[i];

	// SMO loop
	bool complete = (active == n);          // is the current epoch using all points?
	float vio = 0.0f;                       // maximal KKT violation, updated only by thread 0
	uint iter = 0;                          // iteration counter
	while (true)
	{
		__syncthreads();

		// load batchwise data
		if (index < high)
		{
			alpha[threadIdx.x] = a_alpha[index + threadIdx.x];
			uint r = a_row[index + threadIdx.x];
			row[threadIdx.x] = r;
			norm2[threadIdx.x] = G_norm2[r];
			label[threadIdx.x] = a_label[index + threadIdx.x];
			shrink[threadIdx.x] = a_shrink[index + threadIdx.x];
		}
		__syncthreads();

		// loop over points in a batch
		for (uint t=0; t<batchsize && index+t<active; t++)
		{
			// fetch the row of G and compute the inner product with beta using a parallel reduction
			uint r = row[baseindex + t];

			float innerprod = 0.0f;
			#pragma unroll
			for (uint i=threadIdx.x; i<b; i+=z)
			{
				float g = G[r * b + i];
				G_row[i] = g;
				innerprod += g * beta[i];
			}
			__syncthreads();

			// first (inner) warp-level reduction
			innerprod += __shfl_xor_sync(0xffffffff, innerprod, 1);
			innerprod += __shfl_xor_sync(0xffffffff, innerprod, 2);
			innerprod += __shfl_xor_sync(0xffffffff, innerprod, 4);
			innerprod += __shfl_xor_sync(0xffffffff, innerprod, 8);
			innerprod += __shfl_xor_sync(0xffffffff, innerprod, 16);
			if ((threadIdx.x & 31) == 0) intermediate[threadIdx.x / 32] = innerprod;
			__syncthreads();

			// second (outer) warp-level reduction
			if (threadIdx.x < 32)
			{
				innerprod = (32 * threadIdx.x < b) ? intermediate[threadIdx.x] : 0.0f;
				innerprod += __shfl_xor_sync(0xffffffff, innerprod, 1);
				innerprod += __shfl_xor_sync(0xffffffff, innerprod, 2);
				innerprod += __shfl_xor_sync(0xffffffff, innerprod, 4);
				innerprod += __shfl_xor_sync(0xffffffff, innerprod, 8);
				innerprod += __shfl_xor_sync(0xffffffff, innerprod, 16);
			}
			__syncthreads();

			// compute the step, update alpha and the maximal KKT violation (necessarily single threaded)
			if (threadIdx.x == 0)
			{
				float a = alpha[baseindex + t];
				float y = label[baseindex + t];
				float q = norm2[baseindex + t];
				float step = (1.0f - y * innerprod) / q;
				float u = a + step;
				if (u <= 0.0f)
				{
					step = -a;
					alpha[baseindex + t] = 0.0f;
					if (a > vio) vio = a;
				}
				else if (u >= C)
				{
					step = C - a;
					alpha[baseindex + t] = C;
					if (step > vio) vio = step;
				}
				else
				{
					float abs_s = step < 0.0f ? -step : +step;
					alpha[baseindex + t] = u;
					if (abs_s > vio) vio = abs_s;
				}
				shared_step = y * step;
				if (step != 0.0f)
				{
					shrink[baseindex + t] = 0;
				}
				else
				{
					shrink[baseindex + t]++;
				}
			}
			__syncthreads();
			float step = shared_step;

			if (step != 0.0f)
			{
				// perform the step on beta
				#pragma unroll
				for (uint i=threadIdx.x; i<b; i+=z) beta[i] += step * G_row[i];
			}
			else if (shrink[baseindex + t] >= shrink_threshold)
			{
				// shrink the variable, i.e., swap with the last active one
				active--;

				if (active < high)
				{
					// store "high" buffer segments
					if (high + threadIdx.x < n)
					{
						a_alpha[high + threadIdx.x] = alphaH[threadIdx.x];
						a_row[high + threadIdx.x] = rowH[threadIdx.x];
						a_label[high + threadIdx.x] = labelH[threadIdx.x];
//						a_shrink[high + threadIdx.x] = shrinkH[threadIdx.x];   // (unnecessary)
					}
					__syncthreads();

					// move to previous segment
					high -= batchsize;

					// load "high" buffer segment
					if (index < high)
					{
						alphaH[threadIdx.x] = a_alpha[high + threadIdx.x];
						uint r = a_row[high + threadIdx.x];
						rowH[threadIdx.x] = r;
						norm2H[threadIdx.x] = G_norm2[r];
						labelH[threadIdx.x] = a_label[high + threadIdx.x];
						shrinkH[threadIdx.x] = a_shrink[high + threadIdx.x];
					}
					else
					{
						alphaH[threadIdx.x] = alpha[threadIdx.x];
						rowH[threadIdx.x] = row[threadIdx.x];
						norm2H[threadIdx.x] = norm2[threadIdx.x];
						labelH[threadIdx.x] = label[threadIdx.x];
						shrinkH[threadIdx.x] = shrink[threadIdx.x];
					}
					__syncthreads();
				}

				// swap variables
				if (index + t != active)
				{
					if (threadIdx.x == 0)
					{
						SWAP(norm2[baseindex + t], norm2H[active - high])
						SWAP(alpha[baseindex + t], alphaH[active - high])
						if (z <= 64) SWAP(label[baseindex + t], labelH[active - high])
					}
					else if (threadIdx.x == 32)
					{
						SWAP(row[baseindex + t], rowH[active - high])
						if (z <= 96) shrink[baseindex + t] = shrinkH[active - high];
					}
					else if (z > 64 && threadIdx.x == 64) SWAP(label[baseindex + t], labelH[active - high])
					else if (z > 96 && threadIdx.x == 96) shrink[baseindex + t] = shrinkH[active - high];
				}

				// process the swapped point next
				t--;
			}

			unshrink_counter++;
			iter++;
		}

		// store batchwise data
		__syncthreads();
		if (index < high)
		{
			a_alpha[index + threadIdx.x] = alpha[threadIdx.x];
			a_row[index + threadIdx.x] = row[threadIdx.x];
			a_label[index + threadIdx.x] = label[threadIdx.x];
			a_shrink[index + threadIdx.x] = shrink[threadIdx.x];
		}

		// progress to the next batch
		__syncthreads();
		index += batchsize;
		if (index >= active)   // end of the epoch
		{
			epoch++;

			// check stopping criterion
			if (threadIdx.x == 0) shared_vio = vio;
			__syncthreads();
			if (shared_vio <= epsilon)
			{
				if (complete)
				{
					// solution is optimal => stop
					if (threadIdx.x == 0) a_problem[blockIdx.x].solved = true;
					break;
				}
				else
				{
					// force complete unshrinking
					unshrink_counter = 4 * n;
				}
			}

			__syncthreads();
			if (unshrink_counter >= 4 * n)
			{
				// unshrink, i.e., declare that all points are active
				uint newhigh = ((n-1) / batchsize) * batchsize;

				// store/load "high" buffer segments; set shrink counters where appropriate
				if (high < newhigh)
				{
					// store high buffer segments
					a_alpha[high + threadIdx.x] = alphaH[threadIdx.x];
					a_row[high + threadIdx.x] = rowH[threadIdx.x];
					a_label[high + threadIdx.x] = labelH[threadIdx.x];
					a_shrink[high + threadIdx.x] = (high + threadIdx.x >= active) ? shrink_threshold - 1 : shrinkH[threadIdx.x];
					__syncthreads();

					// set unbuffered counters
					for (uint i=high+batchsize+threadIdx.x; i<newhigh; i+=z) a_shrink[i] = shrink_threshold - 1;
					__syncthreads();

					// shift high buffer location
					high = newhigh;

					// load high buffer segments
					if (newhigh + threadIdx.x < n)
					{
						alphaH[threadIdx.x] = a_alpha[newhigh + threadIdx.x];
						uint r = a_row[newhigh + threadIdx.x];
						rowH[threadIdx.x] = r;
						norm2H[threadIdx.x] = G_norm2[r];
						labelH[threadIdx.x] = a_label[newhigh + threadIdx.x];
						shrinkH[threadIdx.x] = shrink_threshold - 1;
					}
					__syncthreads();
				}
				else
				{
					if (high + threadIdx.x >= active && high + threadIdx.x < n) shrinkH[threadIdx.x] = shrink_threshold - 1;
					__syncthreads();
				}

				active = n;
				unshrink_counter = 0;
			}

			if (iter >= maxiter) break;   // stop due to iteration limit
			index = 0;
			complete = (active == n);
			vio = 0.0f;
		}
	}

	// store "high" buffer segments
	__syncthreads();
	if (high + threadIdx.x < n)
	{
		a_alpha[high + threadIdx.x] = alphaH[threadIdx.x];
		a_row[high + threadIdx.x] = rowH[threadIdx.x];
		a_label[high + threadIdx.x] = labelH[threadIdx.x];
		a_shrink[high + threadIdx.x] = shrinkH[threadIdx.x];
	}

	// store beta
	#pragma unroll
	for (uint i=threadIdx.x; i<b; i+=z) a_beta[i] = beta[i];

	// store the solver status
	if (threadIdx.x == 0)
	{
		a_problem[blockIdx.x].active = active;
		a_problem[blockIdx.x].violation = shared_vio;
		a_problem[blockIdx.x].epoch = epoch;
		a_problem[blockIdx.x].iterations += iter;
		a_problem[blockIdx.x].unshrink_counter = unshrink_counter;
	}

#undef batchsize
#undef alpha
#undef alphaH
#undef row
#undef rowH
#undef norm2
#undef norm2H
#undef shrink
#undef shrinkH
#undef beta
#undef G_row
#undef shared_step
#undef shared_vio
#undef SWAP
#undef shrink_threshold
}

// convenience macros for invoking the kernel template
#define CALL(b, z) \
		kernel_cuda_smo<b, z><<<open, z, s>>>( \
				iter, \
				G.data(), \
				norm2.data(), \
				row.data(), \
				y.data(), \
				a_shrink.data(), \
				alpha.data(), \
				beta.data(), \
				a_problem.data() \
			)
#define CASE(b) case b: \
	cudaFuncSetAttribute(&kernel_cuda_smo<b, b<=1024 ? b : 1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, s); \
	CALL(b, b<=1024 ? b : 1024); \
	break;

//
// SMO loop on the GPU. This function repeatedly calls the SMO kernel,
// which solves all problems in the range begin:end in parallel, each
// in one thread block. Problems are removed as soon as they are solved.
//
// The function expects n_sub, offset, and C to cover the whole range of
// problems, while alpha, beta, row and y are restricted to the range
// begin:end.
//
void cuda_smo(uint device, uint b, uint n, uint begin, uint end, vector<uint> const& n_sub, vector<uint> const& offset, DeviceArray<float> const& G, DeviceArray<float> const& norm2, DeviceArray<uint>& row, DeviceArray<float>& y, DeviceArray<float>& alpha, DeviceArray<float>& beta, vector<float> const& C, float epsilon, bool warmstart)
{
	uint active = end - begin;
	uint problems = n_sub.size();
	uint variables = y.size();

	assert(offset.size() == problems);
	assert(G.size() == n * b);
	assert(norm2.size() == n);
	assert(row.size() == variables);
	assert(beta.size() == active * b);
	assert(C.size() == problems);

	vector<Problem> h_problem(active);
	uint n_max = 0;
	for (uint i=0; i<active; i++)
	{
		uint nn = n_sub[begin + i];
		n_max = std::max(n_max, nn);

		h_problem[i].alpha_offset = offset[begin + i] - offset[begin];
		h_problem[i].beta_offset = b * i;
		h_problem[i].n = n_sub[begin + i];
		h_problem[i].active = nn;
		h_problem[i].C = C[begin + i];
		h_problem[i].epsilon = epsilon;
		h_problem[i].epoch = 0;
		h_problem[i].iterations = 0;
		h_problem[i].unshrink_counter = 0;
		h_problem[i].solved = 0;
	}
	DeviceArray<Problem> a_problem(h_problem);

	uint z = std::min<uint>(b, 1024);
	uint s = 40*z + 8*b + 256;
	if (s > cudaDevices[device].max_shared_memory_size)
	{
		uint max_b = ((cudaDevices[device].max_shared_memory_size - 40 * 1024 - 256) / 8) & ~31;
		throwError("Too much shared memory used, exceeding the GPU implementation limit. Try reducing the budget size or processing the data on the CPU. Shared memory limit of CUDA device " + to_string(device) + ": " + to_string(cudaDevices[device].max_shared_memory_size) + " bytes, corresponding to an effective dimension of at most " + to_string(max_b) + ".");
	}

	// call the kernel in a loop
	DeviceArray<uint> a_shrink(variables, true);
	uint open = active;
	uint64_t iter = n;
	for (uint calls=0; open>0; calls++)
	{
		switch (b)
		{
			CASE(64)
			CASE(96)
			CASE(128)
			CASE(160)
			CASE(192)
			CASE(224)
			CASE(256)
			CASE(288)
			CASE(320)
			CASE(352)
			CASE(384)
			CASE(416)
			CASE(448)
			CASE(480)
			CASE(512)
			CASE(544)
			CASE(576)
			CASE(608)
			CASE(640)
			CASE(672)
			CASE(704)
			CASE(736)
			CASE(768)
			CASE(800)
			CASE(832)
			CASE(864)
			CASE(896)
			CASE(928)
			CASE(960)
			CASE(992)
			CASE(1024)
			CASE(1056)
			CASE(1088)
			CASE(1120)
			CASE(1152)
			CASE(1184)
			CASE(1216)
			CASE(1248)
			CASE(1280)
			CASE(1312)
			CASE(1344)
			CASE(1376)
			CASE(1408)
			CASE(1440)
			CASE(1472)
			CASE(1504)
			CASE(1536)
			CASE(1568)
			CASE(1600)
			CASE(1632)
			CASE(1664)
			CASE(1696)
			CASE(1728)
			CASE(1760)
			CASE(1792)
			CASE(1824)
			CASE(1856)
			CASE(1888)
			CASE(1920)
			CASE(1952)
			CASE(1984)
			CASE(2016)
			CASE(2048)
			CASE(2080)
			CASE(2112)
			CASE(2144)
			CASE(2176)
			CASE(2208)
			CASE(2240)
			CASE(2272)
			CASE(2304)
			CASE(2336)
			CASE(2368)
			CASE(2400)
			CASE(2432)
			CASE(2464)
			CASE(2496)
			CASE(2528)
			CASE(2560)
			CASE(2592)
			CASE(2624)
			CASE(2656)
			CASE(2688)
			CASE(2720)
			CASE(2752)
			CASE(2784)
			CASE(2816)
			CASE(2848)
			CASE(2880)
			CASE(2912)
			CASE(2944)
			CASE(2976)
			CASE(3008)
			CASE(3040)
			CASE(3072)
			CASE(3104)
			CASE(3136)
			CASE(3168)
			CASE(3200)
			CASE(3232)
			CASE(3264)
			CASE(3296)
			CASE(3328)
			CASE(3360)
			CASE(3392)
			CASE(3424)
			CASE(3456)
			CASE(3488)
			CASE(3520)
			CASE(3552)
			CASE(3584)
			CASE(3616)
			CASE(3648)
			CASE(3680)
			CASE(3712)
			CASE(3744)
			CASE(3776)
			CASE(3808)
			CASE(3840)
			CASE(3872)
			CASE(3904)
			CASE(3936)
			CASE(3968)
			CASE(4000)
			CASE(4032)
			CASE(4064)
			CASE(4096)
			CASE(4128)
			CASE(4160)
			CASE(4192)
			CASE(4224)
			CASE(4256)
			CASE(4288)
			CASE(4320)
			CASE(4352)
			CASE(4384)
			CASE(4416)
			CASE(4448)
			CASE(4480)
			CASE(4512)
			CASE(4544)
			CASE(4576)
			CASE(4608)
			CASE(4640)
			CASE(4672)
			CASE(4704)
			CASE(4736)
			CASE(4768)
			CASE(4800)
			CASE(4832)
			CASE(4864)
			CASE(4896)
			CASE(4928)
			CASE(4960)
			CASE(4992)
			CASE(5024)
			CASE(5056)
			CASE(5088)
			CASE(5120)
			CASE(5152)
			CASE(5184)
			CASE(5216)
			CASE(5248)
			CASE(5280)
			CASE(5312)
			CASE(5344)
			CASE(5376)
			CASE(5408)
			CASE(5440)
			CASE(5472)
			CASE(5504)
			CASE(5536)
			CASE(5568)
			CASE(5600)
			CASE(5632)
			CASE(5664)
			CASE(5696)
			CASE(5728)
			CASE(5760)
			CASE(5792)
			CASE(5824)
			CASE(5856)
			CASE(5888)
			CASE(5920)
			CASE(5952)
			CASE(5984)
			CASE(6016)
			CASE(6048)
			CASE(6080)
			CASE(6112)
			CASE(6144)
			CASE(6176)
			CASE(6208)
			CASE(6240)
			CASE(6272)
			CASE(6304)
			CASE(6336)
			CASE(6368)
			CASE(6400)
			CASE(6432)
			CASE(6464)
			CASE(6496)
			CASE(6528)
			CASE(6560)
			CASE(6592)
			CASE(6624)
			CASE(6656)
			CASE(6688)
			CASE(6720)
			CASE(6752)
			CASE(6784)
			CASE(6816)
			CASE(6848)
			CASE(6880)
			CASE(6912)
			CASE(6944)
			CASE(6976)
			CASE(7008)
			CASE(7040)
			CASE(7072)
			CASE(7104)
			CASE(7136)
			CASE(7168)
			CASE(7200)
			CASE(7232)
			CASE(7264)
			CASE(7296)
			CASE(7328)
			CASE(7360)
			CASE(7392)
			CASE(7424)
			CASE(7456)
			CASE(7488)
			CASE(7520)
			CASE(7552)
			CASE(7584)
			CASE(7616)
			CASE(7648)
			CASE(7680)
			CASE(7712)
			CASE(7744)
			CASE(7776)
			CASE(7808)
			CASE(7840)
			CASE(7872)
			CASE(7904)
			CASE(7936)
			CASE(7968)
			CASE(8000)
			CASE(8032)
			CASE(8064)
			CASE(8096)
			CASE(8128)
			CASE(8160)
			CASE(8192)
			CASE(8224)
			CASE(8256)
			CASE(8288)
			CASE(8320)
			CASE(8352)
			CASE(8384)
			CASE(8416)
			CASE(8448)
			CASE(8480)
			CASE(8512)
			CASE(8544)
			CASE(8576)
			CASE(8608)
			CASE(8640)
			CASE(8672)
			CASE(8704)
			CASE(8736)
			CASE(8768)
			CASE(8800)
			CASE(8832)
			CASE(8864)
			CASE(8896)
			CASE(8928)
			CASE(8960)
			CASE(8992)
			CASE(9024)
			CASE(9056)
			CASE(9088)
			CASE(9120)
			CASE(9152)
			CASE(9184)
			CASE(9216)
			CASE(9248)
			CASE(9280)
			CASE(9312)
			CASE(9344)
			CASE(9376)
			CASE(9408)
			CASE(9440)
			CASE(9472)
			CASE(9504)
			CASE(9536)
			CASE(9568)
			CASE(9600)
			CASE(9632)
			CASE(9664)
			CASE(9696)
			CASE(9728)
			CASE(9760)
			CASE(9792)
			CASE(9824)
			CASE(9856)
			CASE(9888)
			CASE(9920)
			CASE(9952)
			CASE(9984)
			CASE(10016)
			CASE(10048)
			CASE(10080)
			CASE(10112)
			CASE(10144)
			CASE(10176)
			CASE(10208)
			CASE(10240)
			CASE(10272)
			CASE(10304)
			CASE(10336)
			CASE(10368)
			CASE(10400)
			CASE(10432)
			CASE(10464)
			CASE(10496)
			CASE(10528)
			CASE(10560)
			CASE(10592)
			CASE(10624)
			CASE(10656)
			CASE(10688)
			CASE(10720)
			CASE(10752)
			CASE(10784)
			CASE(10816)
			CASE(10848)
			CASE(10880)
			CASE(10912)
			CASE(10944)
			CASE(10976)
			CASE(11008)
			CASE(11040)
			CASE(11072)
			CASE(11104)
			CASE(11136)
			CASE(11168)
			CASE(11200)
			CASE(11232)
			CASE(11264)
			CASE(11296)
			CASE(11328)
			CASE(11360)
			CASE(11392)
			CASE(11424)
			CASE(11456)
			CASE(11488)
			CASE(11520)
			CASE(11552)
			CASE(11584)
			CASE(11616)
			CASE(11648)
			CASE(11680)
			CASE(11712)
			CASE(11744)
			CASE(11776)
			CASE(11808)
			CASE(11840)
			CASE(11872)
			CASE(11904)
			CASE(11936)
			CASE(11968)
			CASE(12000)
			CASE(12032)
			CASE(12064)
			CASE(12096)
			CASE(12128)
			CASE(12160)
			CASE(12192)
			CASE(12224)
			CASE(12256)
			CASE(12288)
			CASE(12320)
			CASE(12352)
			CASE(12384)
			CASE(12416)
			CASE(12448)
			CASE(12480)
			CASE(12512)
			CASE(12544)
			CASE(12576)
			CASE(12608)
			CASE(12640)
			CASE(12672)
			CASE(12704)
			CASE(12736)
			CASE(12768)
			CASE(12800)
			CASE(12832)
			CASE(12864)
			CASE(12896)
			CASE(12928)
			CASE(12960)
			CASE(12992)
			CASE(13024)
			CASE(13056)
			CASE(13088)
			CASE(13120)
			CASE(13152)
			CASE(13184)
			CASE(13216)
			CASE(13248)
			CASE(13280)
			CASE(13312)
			CASE(13344)
			CASE(13376)
			CASE(13408)
			CASE(13440)
			CASE(13472)
			CASE(13504)
			CASE(13536)
			CASE(13568)
			CASE(13600)
			CASE(13632)
			CASE(13664)
			CASE(13696)
			CASE(13728)
			CASE(13760)
			CASE(13792)
			CASE(13824)
			CASE(13856)
			CASE(13888)
			CASE(13920)
			CASE(13952)
			CASE(13984)
			CASE(14016)
			CASE(14048)
			CASE(14080)
			CASE(14112)
			CASE(14144)
			CASE(14176)
			CASE(14208)
			CASE(14240)
			CASE(14272)
			CASE(14304)
			CASE(14336)
			CASE(14368)
			CASE(14400)
			CASE(14432)
			CASE(14464)
			CASE(14496)
			CASE(14528)
			CASE(14560)
			CASE(14592)
			CASE(14624)
			CASE(14656)
			CASE(14688)
			CASE(14720)
			CASE(14752)
			CASE(14784)
			CASE(14816)
			CASE(14848)
			CASE(14880)
			CASE(14912)
			CASE(14944)
			CASE(14976)
			CASE(15008)
			CASE(15040)
			CASE(15072)
			CASE(15104)
			CASE(15136)
			CASE(15168)
			CASE(15200)
			CASE(15232)
			CASE(15264)
			CASE(15296)
			CASE(15328)
			CASE(15360)
			CASE(15392)
			CASE(15424)
			CASE(15456)
			CASE(15488)
			CASE(15520)
			CASE(15552)
			CASE(15584)
			CASE(15616)
			CASE(15648)
			CASE(15680)
			CASE(15712)
			CASE(15744)
			CASE(15776)
			CASE(15808)
			CASE(15840)
			CASE(15872)
			CASE(15904)
			CASE(15936)
			CASE(15968)
			CASE(16000)
			CASE(16032)
			CASE(16064)
			CASE(16096)
			CASE(16128)
			CASE(16160)
			CASE(16192)
			CASE(16224)
			CASE(16256)
			CASE(16288)
			CASE(16320)
			CASE(16352)
			CASE(16384)
			default:
				if (b >= 64 && b <= 16384 && (b & 31) == 0) throwError("internal error");
				else throwError("The feature dimension is not a multiple of 32 or it is smaller than 64 or larger than 16384.");
		}
		CheckKernelLaunch;

		a_problem.to_cpu(h_problem);

		// remove completed problems
		bool changed = false;
		for (uint i=0; i<open; i++)
		{
			if (h_problem[i].solved != 0) changed = true;
		}
		if (changed)
		{
			for (uint i=0; i<open; i++)
			{
				if (h_problem[i].solved != 0)
				{
					open--;
					if (i != open) h_problem[i] = h_problem[open];
					i--;
					ProgressBar::increment();
				}
			}
			a_problem.to_gpu(h_problem);
		}
	}

	if (warmstart)
	{
		// sort alpha back into the original order
		DeviceArray<uint> gpu_offset(end-begin, &offset[begin]);
		DeviceArray<uint> gpu_n_sub(end-begin, &n_sub[begin]);
		cuda_sort(offset[begin], gpu_offset, gpu_n_sub, n_max, row, alpha);
	}
}
