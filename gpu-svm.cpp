
//
// Functions for preparation, training and prediction on the GPU.
//


#include "definitions.h"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <tuple>

using namespace std;


namespace gpu {


//
// Prepare the matrices G and V describing the feature space.
// The function returns the compressed dimension b, the decompression
// matrix V, the compressed kernel matrix G, and its row-wise squared
// norms.
// This version of the function is designed to run on a single GPU. It
// is supposed to be called from within a GPU handling thread.
//
tuple<uint, DeviceArray<float>, DeviceArray<float>, DeviceArray<float>> prepare_for_gpu(uint device, SparseMatrix const& x, GpuSparseMatrix const& basis, Kernel const& kernel, uint budget)
{
	// dataset size
	uint n = x.rows();

	cout << "kernel feature space projection for " << kernel.toString() << " ..." << flush;
	WallClockTimer timer;

	// compute the inverse factor of the kernel matrix of the basis, projected using PCA
	DeviceArray<float> gpu_K_V = cuda_kernelmatrix(basis, basis, kernel);
	uint b = cuda_inv_factor(device, budget, gpu_K_V);
	cout << " effective dimension: " << b << "; completed in " << timer.seconds() << " seconds" << endl;

	// compute the compressed kernel matrix G
	timer.restart();
	DeviceArray<float> gpu_G(n * b, false);
	uint begin = 0;
	ProgressBar::start(n, "kernel matrix computation");
	while (begin < n)
	{
		// 64M chunks
		uint64_t mem = 0;
		uint end = begin;
		while (end < n)
		{
			uint64_t rowmem = 8 * x.nnz(end) + 4 + 4 * budget + 4 * b;
			if (end > begin && mem + rowmem >= 0x4000000) break;
			mem += rowmem;
			end++;
		}

		// load the chunk into GPU memory
		GpuSparseMatrix gpu_x(x, begin, end);

		// compute the corresponding block of the compressed kernel matrix G
		DeviceArray<float> gpu_temp = cuda_kernelmatrix(gpu_x, basis, kernel);
		cuda_matmul(device, end-begin, budget, b, gpu_temp.data(), gpu_K_V.data(), gpu_G.data() + begin * b, budget, b);

		// move on to the next block
		ProgressBar::increment(end - begin);
		begin = end;
	}

	// compute row-wise squared norms
	DeviceArray<float> gpu_norm2 = cuda_rows_norm2(n, gpu_G);

	cuda_sync();
	ProgressBar::stop();
	cout << "kernel matrix computation completed in " << timer.seconds() << " seconds" << endl;

	return make_tuple(b, std::move(gpu_K_V), std::move(gpu_G), std::move(gpu_norm2));
}

//
// Prepare the matrices G and V describing the feature space.
// The function returns the compressed dimension b, the decompression
// matrix V, the compressed kernel matrix G, and its row-wise squared
// norms.
//
tuple<uint, MatrixCM, MatrixRM, vector<float>> prepare_for_cpu(SparseMatrix const& x, unordered_map<uint, GpuSparseMatrix> const& basis, Kernel const& kernel, uint budget)
{
	// dataset size
	uint n = x.rows();

	cout << "kernel feature space projection for " << kernel.toString() << " ..." << flush;
	WallClockTimer timer;

	// compute the inverse factor of the kernel matrix of the basis, projected using PCA
	MatrixCM V;
	uint b = 0;
	WorkerPool::enqueue(
		[&](uint id)
		{
			DeviceArray<float> gpu_K_V = cuda_kernelmatrix(basis.at(id), basis.at(id), kernel);
			b = cuda_inv_factor(id, budget, gpu_K_V);
			V.create(budget, b);
			gpu_K_V.to_cpu(V.values);
		}
	);
	WorkerPool::wait();

	// distribute the data to all GPUs
	unordered_map<uint, DeviceArray<float>> gpu_V;
	for (uint id : WorkerPool::workerIDs()) gpu_V[id] = DeviceArray<float>();
	WorkerPool::broadcast([&](uint id)
		{
			gpu_V[id].create(V.values.size(), V.data());
		});
	WorkerPool::wait();
	cout << " effective dimension: " << b << "; completed in " << timer.seconds() << " seconds" << endl;

	// compute the kernel matrix in chunks
	timer.restart();
	uint64_t chunksize = std::min<uint64_t>(0x4000000, 8 * x.nnz() / WorkerPool::size());
	MatrixRM G(n, b);
	vector<float> norm2(n);
	uint begin = 0;
	ProgressBar::start(n, "kernel matrix computation");
	while (begin < n)
	{
		// 64M chunks
		uint64_t mem = 0;
		uint end = begin;
		while (end < n)
		{
			uint64_t rowmem = 8 * x.nnz(end) + 4 + 4 * budget + 4 * b;
			if (end > begin && mem + rowmem >= chunksize) break;
			mem += rowmem;
			end++;
		}

		WorkerPool::enqueue(
			[begin, end, b, budget, &x, &basis, &G, &kernel, &gpu_V, &norm2](uint id)
			{
				// load the chunk into GPU memory
				GpuSparseMatrix gpu_x(x, begin, end);

				// compute the corresponding block of the compressed kernel matrix G
				DeviceArray<float> gpu_temp = cuda_kernelmatrix(gpu_x, basis.at(id), kernel);
				DeviceArray<float> gpu_G((end-begin) * b, false);
				cuda_matmul(id, end-begin, budget, b, gpu_temp, gpu_V[id], gpu_G, "RCR");
				gpu_G.to_cpu(&G(begin, 0));

				// compute row-wise squared norms
				DeviceArray<float> gpu_norm2 = cuda_rows_norm2(end-begin, gpu_G);
				gpu_norm2.to_cpu(&norm2[begin]);

				// move on to the next block
				ProgressBar::increment(end - begin);
			}
		);

		begin = end;
	}
	WorkerPool::broadcastLast([](uint id) { cuda_sync(); });
	WorkerPool::wait();
	ProgressBar::stop();
	cout << "kernel matrix computation completed in " << timer.seconds() << " seconds" << endl;

	return make_tuple(b, std::move(V), std::move(G), norm2);
}

//
// Train an SVM. The function fills in the alpha matrix in the model.
// This version of the function is designed to run on a single GPU. It
// is supposed to be called from within a GPU handling thread.
//
void train(uint device, DeviceArray<float> const& V, DeviceArray<float> const& G, DeviceArray<float> const& norm2, vector<float> const& labels, Model& model, float C, float epsilon)
{
	WallClockTimer timer;

	uint n = labels.size();          // number of data points
	uint b = G.size() / n;           // compressed weight dimension
	uint budget = V.size() / b;      // uncompressed weight dimension
	uint c = model.classes.size();   // number of classes
	uint p = c * (c-1) / 2;          // number of pairs of classes

	ProgressBar::start(p, "SVM training");

	// prepare problem definitions
	vector<uint> n_sub(p);
	vector<uint> offset(p);
	vector<float> C_sub(p);
	vector<float> y; y.reserve((c-1) * n);
	vector<uint> row; row.reserve((c-1) * n);
	for (uint i=0, j=1, k=0; k<p; k++)
	{
		offset[k] = row.size();
		for (uint l=0; l<n; l++)
		{
			if (labels[l] == model.classes[i] || labels[l] == model.classes[j])
			{
				y.push_back(labels[l] == model.classes[i] ? -1.0f : +1.0f);
				row.push_back(l);
			}
		}
		n_sub[k] = row.size() - offset[k];
		C_sub[k] = C;
		j++; if (j == c) { i++; j = i+1; }
	}
	assert(y.size() == (c-1) * n);
	assert(row.size() == (c-1) * n);

	// call the CUDA solver
	DeviceArray<uint> gpu_row(row);
	DeviceArray<float> gpu_y(y);
	DeviceArray<float> gpu_alpha((c-1) * n, true);
	DeviceArray<float> gpu_beta(p * b, true);
	cuda_smo(device, b, n, 0, p, n_sub, offset, G, norm2, gpu_row, gpu_y, gpu_alpha, gpu_beta, C_sub, epsilon, false);

	// fill in alpha = V * beta
	model.alpha.create(budget, p);
	model.gpu_alpha[device].create(budget * p, false);
	cuda_matmul(device, budget, b, p, V, gpu_beta, model.gpu_alpha[device], "CCC");
	model.gpu_alpha[device].to_cpu(model.alpha.data());

	cuda_sync();

	ProgressBar::stop();
	cout << "SVM training completed in " << timer.seconds() << " seconds" << endl;
}

//
// Train an SVM. The function fills in the alpha matrix in the model.
//
void train(MatrixCM const& V, MatrixRM const& G, vector<float> const& norm2, vector<float> const& labels, Model& model, float C, float epsilon)
{
	uint n = G.rows;
	uint budget = V.rows;
	uint b = G.cols;
	uint c = model.classes.size();
	uint p = c * (c - 1) / 2;
	assert(labels.size() == n);
	assert(V.cols == b);

	WallClockTimer timer;
	ProgressBar::start(p, "SVM training");

	// prepare binary problems
	vector<uint> n_sub(p);
	vector<uint> offset(p);
	vector<float> C_sub(p, C);
	vector<uint> yindex(n);
	{
		// prepare label assignment and class sizes
		unordered_map<float, uint> label2index;
		for (uint i=0; i<c; i++) label2index[model.classes[i]] = i;
		vector<uint> class_n(c, 0);
		for (uint l=0; l<n; l++)
		{
			uint i = label2index[labels[l]];
			yindex[l] = i;
			class_n[i]++;
		}

		// prepare problem sizes and start indices
		uint o = 0;
		for (uint i=0, j=1, k=0; k<p; k++)
		{
			offset[k] = o;
			uint m = class_n[i] + class_n[j];
			n_sub[k] = m;
			o += m;
			j++; if (j == c) { i++; j = i+1; }
		}
	}

	// transfer data to all GPUs
	unordered_map<uint, DeviceArray<uint>> gpu_offset;
	unordered_map<uint, DeviceArray<uint>> gpu_yindex;
	unordered_map<uint, DeviceArray<float>> gpu_V;
	unordered_map<uint, DeviceArray<float>> gpu_G;
	unordered_map<uint, DeviceArray<float>> gpu_norm2;
	for (uint id : WorkerPool::workerIDs())
	{
		gpu_offset[id] = DeviceArray<uint>();
		gpu_yindex[id] = DeviceArray<uint>();
		gpu_V[id] = DeviceArray<float>();
		gpu_G[id] = DeviceArray<float>();
		gpu_norm2[id] = DeviceArray<float>();
	}
	WorkerPool::broadcast([&](uint id)
		{
			gpu_offset[id].create(offset);
			gpu_yindex[id].create(yindex);
			gpu_V[id].create(V.values.size(), V.data());
			gpu_G[id].create(G.values.size(), G.data());
			gpu_norm2[id].create(norm2);
			cuda_sync();
		});

	// actual SVM training
	model.alpha.create(budget, p);
	uint maxchunksize = std::min<std::size_t>(0x4000000, 8ULL * (offset.back() + n_sub.back()) / WorkerPool::size() + 1);
	for (uint begin=0; begin<p;)
	{
		// 64M chunks
		uint64_t num = 0;
		uint end = begin;
		while (end < p)
		{
			uint64_t sub = n_sub[end];
			num += sub;
			end++;
			if (end > begin && (8 * num >= maxchunksize)) break;
		}

		WorkerPool::enqueue(
			[&, begin, end, num, b, c](uint id)
			{
				// reserve memory
				DeviceArray<uint> gpu_row(num, false);
				DeviceArray<float> gpu_y(num, false);
				DeviceArray<float> gpu_alpha(num, true);                            // complete weight vector
				DeviceArray<float> gpu_beta((end - begin) * b, true);               // compressed weight vector
				DeviceArray<float> gpu_alpha_prime((end - begin) * budget, true);   // budgeted weight vector

				// prepare binary problems
				cuda_prepare_binary(begin, end, c, gpu_offset[id], gpu_yindex[id], gpu_row, gpu_y);

				// call the CUDA solver
				cuda_smo(id, b, n, begin, end, n_sub, offset, gpu_G[id], gpu_norm2[id], gpu_row, gpu_y, gpu_alpha, gpu_beta, C_sub, epsilon, false);

				// fill in alpha = V * beta
				cuda_matmul(id, budget, b, end-begin, gpu_V[id], gpu_beta, gpu_alpha_prime, "CCC");

				// copy beta back to the CPU
				gpu_alpha_prime.to_cpu(model.alpha.data() + budget * begin);
			}
		);

		// move on to the next block
		begin = end;
	}
	WorkerPool::broadcastLast([](uint id) { cuda_sync(); });
	WorkerPool::wait();
	ProgressBar::stop();
	cout << "SVM training completed in " << timer.seconds() << " seconds" << endl;
}

//
// Train SVMs using cross-validation. The function returns the
// compressed weight matrix beta, with one block per fold and per value
// of C. It does not fill in the alpha matrix in the model.
//
MatrixCM train_cv(unordered_map<uint, DeviceArray<float>> const& G, vector<float> const& norm2, vector<float> const& labels, Model& model, uint folds, Range const& C, float epsilon, bool warmstart)
{
	uint n = norm2.size();
	uint b = G.begin()->second.size() / n;
	uint c = model.classes.size();
	uint p = c * (c - 1) / 2;
	uint Cs = C.size();
	uint total = p * folds * Cs;
	assert(labels.size() == n);

	// enable warm starts if memory permits
	uint64_t required = 4 * (uint64_t)n * (folds - 1) * C.size();
	uint64_t available = availableRAM();
	if (4 * required > 3 * available) warmstart = false;
	vector<float> alpha;               // readily trained alpha vectors
	vector<bool> done(total, false);   // indicator: is a particular solution ready?
	mutex done_mutex;                  // mutex protecting "done"
	size_t period = p * folds;

	WallClockTimer timer;
	ProgressBar::start(total, warmstart ? "cross-validation SVM training using warm-starts" : "cross-validation SVM training using cold-starts");

	// prepare binary problems
	vector<uint> n_sub(total);
	vector<uint> offset(total);
	vector<float> C_sub(total);
	vector<uint> yindex(n);
	{
		// prepare label assignment and class sizes
		unordered_map<float, uint> label2index;
		for (uint i=0; i<c; i++) label2index[model.classes[i]] = i;
		vector<vector<uint>> class_n(c, vector<uint>(folds, 0));
		for (uint l=0; l<n; l++)
		{
			uint i = label2index[labels[l]];
			yindex[l] = i;
			for (uint f=0; f<folds; f++) if (l % folds != f) class_n[i][f]++;
		}

		// prepare problem sizes and start indices
		uint z = 0, o = 0;
		for (uint s=0; s<Cs; s++)
		{
			for (uint f=0; f<folds; f++)
			{
				for (uint i=0, j=1; i<c-1; )
				{
					offset[z] = o;
					uint m = class_n[i][f] + class_n[j][f];
					n_sub[z] = m;
					C_sub[z] = C[s];
					o += m; z++;
					j++; if (j == c) { i++; j = i+1; }
				}
			}
		}

		// reserve memory for alpha vectors
		if (warmstart) alpha.resize(o);
	}

	// transfer data to all GPUs
	unordered_map<uint, DeviceArray<uint>> gpu_offset;
	unordered_map<uint, DeviceArray<uint>> gpu_yindex;
	unordered_map<uint, DeviceArray<float>> gpu_norm2;
	for (uint id : WorkerPool::workerIDs())
	{
		gpu_offset[id] = DeviceArray<uint>();
		gpu_yindex[id] = DeviceArray<uint>();
		gpu_norm2[id] = DeviceArray<float>();
	}
	WorkerPool::broadcast([&](uint id)
		{
			gpu_offset[id].create(offset);
			gpu_yindex[id].create(yindex);
			gpu_norm2[id].create(norm2);
			cuda_sync();
		});

	// actual SVM training
	MatrixCM beta(b, total);
	uint maxchunksize = std::min<std::size_t>(0x4000000, 8ULL * (offset.back() + n_sub.back()) / WorkerPool::size() + 1);
	for (uint begin=0; begin<total;)
	{
		// 64M chunks
		uint64_t num = 0;
		uint end = begin;
		while (end < total)
		{
			uint64_t sub = n_sub[end];
			num += sub;
			end++;
			if (end > begin && (8 * num >= maxchunksize)) break;
		}

		WorkerPool::enqueue(
				[&, begin, end, num, b, c](uint id)
			{
				// reserve memory
				DeviceArray<uint> gpu_row(num, false);
				DeviceArray<float> gpu_y(num, false);
				DeviceArray<float> gpu_alpha(num, true);
				DeviceArray<float> gpu_beta((end - begin) * b, true);

				// prepare binary problems
				cuda_prepare_binary_cv(begin, end, Cs, folds, c, gpu_offset[id], gpu_yindex[id], gpu_row, gpu_y);
				if (warmstart)
				{
					// prepare alpha and beta as copies of completed runs
					lock_guard<mutex> lock(done_mutex);
					for (uint q=begin; q<end; q++)
					{
						int origin = (int)q - (int)period;
						while (origin >= 0)
						{
							assert(n_sub[origin] == n_sub[q]);
							if (done[origin]) break;
							origin -= period;
						}
						if (origin >= 0)
						{
							gpu_alpha.to_gpu(&alpha[offset[origin]], offset[q] - offset[begin], n_sub[q]);
							gpu_beta.to_gpu(&beta(0, origin), (q-begin) * b, b);
						}
					}
				}

				// call the CUDA solver
				cuda_smo(id, b, n, begin, end, n_sub, offset, G.at(id), gpu_norm2[id], gpu_row, gpu_y, gpu_alpha, gpu_beta, C_sub, epsilon, warmstart);

				// copy beta back to the CPU
				gpu_beta.to_cpu(beta.data() + b * begin);

				// copy alpha back to the CPU
				if (warmstart)
				{
					// store alpha for later use
					gpu_alpha.to_cpu(&alpha[offset[begin]]);

					// mark problems as solved
					std::unique_lock<mutex> lock(done_mutex);
					std::fill(done.begin() + begin, done.begin() + end, true);
				}
			}
		);

		// move on to the next block
		begin = end;
	}
	WorkerPool::broadcastLast([](uint id) { cuda_sync(); });
	WorkerPool::wait();

	ProgressBar::stop();
	cout << "cross-validation SVM training completed in " << timer.seconds() << " seconds" << endl;

	return beta;
}

//
// Compute predictions of the SVM model on data.
//
vector<float> predict(Model const& model, SparseMatrix const& data)
{
	WallClockTimer timer;

	size_t n = data.rows();
	size_t budget = model.alpha.rows;
	size_t problems = model.alpha.cols;

	// compute the predictions in chunks
	vector<float> predictions(n);
	ProgressBar::start(n, "computing predictions");
	WorkerPool::wait();
	uint begin = 0;
	while (begin < n)
	{
		// 64M chunks
		uint64_t mem = 0;
		uint end = begin;
		while (end < n)
		{
			uint64_t rowmem = 8 * data.nnz(end) + 4 + 4 * budget + 4 * problems + 4;
			if (end > begin && mem + rowmem >= 0x4000000) break;
			mem += rowmem;
			end++;
		}

		WorkerPool::enqueue([begin, end, problems, budget, &model, &data, &predictions](uint id)
				{
					// load the chunk into GPU memory
					GpuSparseMatrix gpu_data(data, begin, end);

					// compute the corresponding block of the compressed kernel matrix G
					DeviceArray<float> gpu_temp = cuda_kernelmatrix(gpu_data, model.gpu_basis.at(id), model.kernel);
					DeviceArray<float> gpu_result((end-begin) * problems, false);
					cuda_matmul(id, end-begin, budget, problems, gpu_temp, model.gpu_alpha.at(id), gpu_result, "RCR");
					DeviceArray<float> gpu_predictions(end-begin, false);
					cuda_vote(gpu_result, model.gpu_classes.at(id), gpu_predictions);
					gpu_predictions.to_cpu(&predictions[begin]);

					// move on to the next block
					ProgressBar::increment(end - begin);
				}
			);

		begin = end;
	}
	WorkerPool::broadcastLast([](uint id) { cuda_sync(); });
	WorkerPool::wait();
	ProgressBar::stop();
	cout << "predictions completed in " << timer.seconds() << " seconds" << endl;

	return predictions;
}

//
// Compute cross-validation predictions. The predictions are computed
// directly from the matrix G and the compressed weight vector. Hence,
// the function does not need to compute a kernel matrix from data.
//
vector<vector<float>> predict_cv(std::unordered_map<uint, DeviceArray<float>> const& G, std::unordered_map<uint, DeviceArray<float>> const& classes, MatrixCM const& beta, uint folds)
{
	uint c = classes.begin()->second.size();
	uint p = c * (c-1) / 2;
	uint b = beta.rows;
	uint n = G.begin()->second.size() / b;
	uint Cs = beta.cols / (p * folds);

	WallClockTimer timer;
	ProgressBar::start(Cs*n, "computing cross-validation predictions");

	// 256M chunks
	uint maxchunksize = 0x10000000 / (4 * p);
	maxchunksize -= maxchunksize % folds;

	// pre-allocate GPU memory
	unordered_map<uint, DeviceArray<float>> result;
	unordered_map<uint, DeviceArray<float>> predictions;
	unordered_map<uint, DeviceArray<float>> gpu_beta;
	for (uint id : WorkerPool::workerIDs())
	{
		result[id] = DeviceArray<float>();
		predictions[id] = DeviceArray<float>();
		gpu_beta[id] = DeviceArray<float>();
	}
	WorkerPool::broadcast([&](uint id)
		{
			result[id].create(maxchunksize * p, false);
			predictions[id].create(maxchunksize, false);
			gpu_beta[id].create(beta.values.size(), beta.data());
		});

	vector<vector<float>> ret(Cs);
	for (uint C=0; C<Cs; C++)
	{
		ret[C].resize(n);

		// slice this computation up into chunks so everything fits into memory
		for (uint begin=0; begin<n; begin+=maxchunksize)
		{
			uint end = std::min<uint>(begin + maxchunksize, n);
			uint chunksize = end - begin;
			WorkerPool::enqueue(
				[&, C, begin, chunksize, folds, b, p](uint id)
				{
					for (uint f=0; f<folds; f++)
					{
						uint nn = (chunksize - f + folds - 1) / folds;
						cuda_matmul(id, nn, b, p, G.at(id).data() + (begin + f) * b, gpu_beta[id].data() + (C * folds + f) * p * b, result[id].data() + f * p, folds * b, folds * p);
					}
					cuda_vote(result[id], classes.at(id), predictions[id]);
					predictions[id].to_cpu(ret[C].data() + begin, 0, chunksize);
					ProgressBar::increment(chunksize);
				}
			);
		}
	}

	WorkerPool::broadcastLast([](uint id) { cuda_sync(); });
	WorkerPool::wait();
	ProgressBar::stop();
	cout << "cross-validation predictions completed in " << timer.seconds() << " seconds" << endl;

	return ret;
}

//
// Compute cross-validation predictions. The predictions are computed
// directly from the matrix G and the compressed weight vector. Hence,
// the function does not need to compute a kernel matrix from data.
// This version of the function does not assume that the matrix G fits
// into GPU memory.
//
std::vector<std::vector<float>> predict_cv(MatrixRM const& G, std::unordered_map<uint, DeviceArray<float>> const& classes, MatrixCM const& beta, uint folds)
{
	uint c = classes.begin()->second.size();
	uint p = c * (c-1) / 2;
	uint n = G.rows;
	uint b = G.cols;
	uint Cs = beta.cols / (p * folds);
	assert(beta.rows == b);

	WallClockTimer timer;
	ProgressBar::start(Cs*n, "computing cross-validation predictions");

	// 256M chunks
	uint maxchunksize = 0x10000000 / std::max(4 * p, b);
	maxchunksize -= maxchunksize % folds;

	// pre-allocate GPU memory
	unordered_map<uint, DeviceArray<float>> result;
	unordered_map<uint, DeviceArray<float>> predictions;
	unordered_map<uint, DeviceArray<float>> gpu_beta;
	unordered_map<uint, DeviceArray<float>> gpu_G;
	for (uint id : WorkerPool::workerIDs())
	{
		result[id] = DeviceArray<float>();
		predictions[id] = DeviceArray<float>();
		gpu_beta[id] = DeviceArray<float>();
		gpu_G[id] = DeviceArray<float>();
	}
	WorkerPool::broadcast([&](uint id)
		{
			result[id].create(maxchunksize * p, false);
			predictions[id].create(maxchunksize, false);
			gpu_beta[id].create(beta.values.size(), beta.data());
			gpu_G[id].create(maxchunksize * b, false);
		});

	vector<vector<float>> ret(Cs);
	for (uint C=0; C<Cs; C++)
	{
		ret[C].resize(n);

		// slice this computation up into chunks so everything fits into memory
		for (uint begin=0; begin<n; begin+=maxchunksize)
		{
			uint end = std::min<uint>(begin + maxchunksize, n);
			uint chunksize = end - begin;
			WorkerPool::enqueue(
				[&, C, begin, chunksize, folds, b, p](uint id)
				{
					gpu_G[id].to_gpu(&G(begin, 0), 0, chunksize * b);
					for (uint f=0; f<folds; f++)
					{
						uint nn = (chunksize - f + folds - 1) / folds;
						cuda_matmul(id, nn, b, p, gpu_G.at(id).data() + f * b, gpu_beta[id].data() + (C * folds + f) * p * b, result[id].data() + f * p, folds * b, folds * p);
					}
					cuda_vote(result[id], classes.at(id), predictions[id]);
					predictions[id].to_cpu(ret[C].data() + begin, 0, chunksize);
					ProgressBar::increment(chunksize);
				}
			);
		}
	}

	WorkerPool::broadcastLast([](uint id) { cuda_sync(); });
	WorkerPool::wait();
	ProgressBar::stop();
	cout << "cross-validation predictions completed in " << timer.seconds() << " seconds" << endl;

	return ret;
}


};
