
//
// Functions for preparation, training and prediction on the CPU.
//


#include "definitions.h"

// we leave parallelization to OMP:
#define EIGEN_DONT_PARALLELIZE

#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "Eigen/SparseCore"

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <mutex>

using namespace std;


using EMatrixRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EMatrixCM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using RowVec = Eigen::Matrix<float, 1, Eigen::Dynamic>;
using ColVec = Eigen::Matrix<float, Eigen::Dynamic, 1>;
using Data = Eigen::SparseMatrix<float, Eigen::RowMajor>;


// zero-cost adapters
Eigen::Map<Data const> asEigen(SparseMatrix const& data, vector<uint> const& offset, size_t z = 0)
{
	return Eigen::Map<Data const>(offset.size()-1, data.cols, offset.back(), reinterpret_cast<int const*>(offset.data()), reinterpret_cast<int const*>(data.column.data() + z), data.value.data() + z);
}

Eigen::Map<EMatrixRM> asEigen(MatrixRM& matrix)
{
	return Eigen::Map<EMatrixRM>(matrix.data(), matrix.rows, matrix.cols);
}

Eigen::Map<EMatrixRM const> asEigen(MatrixRM const& matrix)
{
	return Eigen::Map<EMatrixRM const>(matrix.data(), matrix.rows, matrix.cols);
}

Eigen::Map<EMatrixCM> asEigen(MatrixCM& matrix)
{
	return Eigen::Map<EMatrixCM>(matrix.data(), matrix.rows, matrix.cols);
}

Eigen::Map<EMatrixCM const> asEigen(MatrixCM const& matrix)
{
	return Eigen::Map<EMatrixCM const>(matrix.data(), matrix.rows, matrix.cols);
}

Eigen::Map<ColVec> asEigen(vector<float>& colvec)
{
	return Eigen::Map<ColVec>(colvec.data(), colvec.size());
}

Eigen::Map<ColVec const> asEigen(vector<float> const& colvec)
{
	return Eigen::Map<ColVec const>(colvec.data(), colvec.size());
}


namespace cpu {


//
// squared euclidean distances (times a factor)
//
template <typename D1, typename D2>
EMatrixRM squaredDistance(D1 const& x1, D2 const& x2, float factor = 1)
{
	uint n1 = x1.rows();
	uint n2 = x2.rows();
	RowVec x1sq(n1);
	for (uint i=0; i<n1; i++) x1sq(0, i) = x1.row(i).squaredNorm();
	RowVec x2sq(n2);
	for (uint i=0; i<n2; i++) x2sq(0, i) = x2.row(i).squaredNorm();
	EMatrixRM ret = (-2.0f * factor) * x1 * x2.transpose();
	for (uint i=0; i<n1; i++)
	{
		ret.row(i) += factor * x2sq + EMatrixRM::Constant(1, n2, factor * x1sq(0, i));
	}
	return ret;
}

//
// inner products (times a factor)
//
template <typename D1, typename D2>
EMatrixRM innerProduct(D1 const& x1, D2 const& x2, float factor = 1, float offset = 0)
{
	EMatrixRM p = x1 * x2.transpose();
	p.array() *= factor;
	p.array() += offset;
	return p;
}

//
// Gaussian kernel computation for batches of data.
//
template <typename D1, typename D2>
EMatrixRM kernelmatrix(D1 const& x1, D2 const& x2, Kernel const& kernel)
{
	if (kernel.type == Kernel::Gaussian) return squaredDistance(x1, x2, -kernel.gamma).array().exp().matrix();
	else if (kernel.type == Kernel::Polynomial) return innerProduct(x1, x2, kernel.gamma, kernel.offset).array().pow(kernel.degree).matrix();
	else if (kernel.type == Kernel::TanH) return innerProduct(x1, x2, kernel.gamma, kernel.offset).array().tanh().matrix();
	else throwError("unknown kernel type");
}

//
// interface to the parallel eigen solver
//
std::tuple<EMatrixRM, EMatrixRM> eig(EMatrixRM const& K);

template <typename T>
void applyPermutation(vector<T>& data, vector<uint> const& permutation)
{
	uint n = permutation.size();
	vector<T> temp(n);
	for (uint i=0; i<n; i++) temp[permutation[i]] = data[i];
	for (uint i=0; i<n; i++) data[i] = temp[i];
}

template <typename T>
void applyPermutation(T* data, vector<uint> const& permutation)
{
	uint n = permutation.size();
	vector<T> temp(n);
	for (uint i=0; i<n; i++) temp[permutation[i]] = data[i];
	for (uint i=0; i<n; i++) data[i] = temp[i];
}

//
// Prepare the matrices G and V describing the feature space.
// The function returns the compressed dimension, the decompression
// matrix V, the compressed kernel matrix G, and its row-wise squared
// norms.
//
tuple<uint, MatrixCM, MatrixRM, vector<float>> prepare(SparseMatrix const& x, SparseMatrix const& basis, Kernel const& kernel, uint budget)
{
	// PCA eigenvalue cutoff factor
	constexpr float cutoff = 1e-6;   // about 8x the numerical limit

	// dataset size
	uint n = x.rows();

	cout << "kernel feature space projection for " << kernel.toString() << " ..." << flush;
	WallClockTimer timer;

	vector<uint> basisoffset(basis.offset.begin(), basis.offset.end());
	auto ebasis = asEigen(basis, basisoffset);

	// projection to the PCA subspace
	EMatrixRM K = kernelmatrix(ebasis, ebasis, kernel);
	EMatrixRM D, U;
	tie(D, U) = eig(K);
	float threshold = cutoff * D.maxCoeff();
	auto begin = &D(0);
	uint b = std::distance(std::lower_bound(begin, begin+budget, threshold), begin+budget);
	b = (b + 31) & (~31);
	if (b > budget) b -= 32;
	MatrixCM V(budget, b);
	auto eV = asEigen(V);
	for (uint i=0; i<b; i++)
	{
		float d = D(i+budget-b);
		if (d > 0) eV.col(i) = U.col(i+budget-b) / sqrt(d);
		else eV.col(i).setZero();
	}
	cout << " effective dimension: " << b << "; completed in " << timer.seconds() << " seconds" << endl;

	// compute the projected kernel matrix
	timer.restart();
	ProgressBar::start(n, "kernel matrix computation");
	MatrixRM G(n, b);
	vector<float> norm2(n);
	auto eG = asEigen(G);
	#pragma omp parallel for
	for (uint i=0; i<n; i+=4096)
	{
		uint j = min(i+4096, n);

		size_t z = x.offset[i];
		vector<uint> offset(j-i+1);
		for (uint k=i; k<=j; k++) offset[k-i] = x.offset[k] - z;
		auto xblock = asEigen(x, offset, z);

		eG.topRows(j).bottomRows(j-i) = kernelmatrix(xblock, ebasis, kernel) * eV;
		for (uint k=i; k<j; k++) norm2[k] = eG.row(k).squaredNorm();
		ProgressBar::increment(j - i);
	}
	ProgressBar::stop();
	cout << "kernel matrix computation completed in " << timer.seconds() << " seconds" << endl;

	return make_tuple(b, V, G, norm2);
}

//
// Train an SVM for binary classification. During training, norm2, rows,
// alpha and labels are permuted. The function supports warm starts from
// a feasible solution. The vectors alpha and beta must be consistent,
// i.e., represent the same solution.
//
void train_binary(MatrixRM const& G, vector<float>& norm2, vector<uint>& rows, vector<float>& labels, float* alpha, float* beta, float C, float epsilon)
{
	// enable/disable shrinking
	constexpr bool shrinking = true;
	constexpr uint8_t shrink_threshold = 5;

	// sizes
	uint b = G.cols;
	uint n = rows.size();
	assert(labels.size() == n);

	// Eigen proxies
	Eigen::Map<ColVec> ebeta(beta, b);
	auto eG = asEigen(G);

	// epoch loop
	uint active = n;
	uint unshrink_counter = 0;
	vector<uint8_t> inactive_counter(n, 0);
	while (true)
	{
		float vio = 0;
		for (uint i=0; i<active; i++)
		{
			// calculate the margin
			float step = (1.0f - labels[i] * eG.row(rows[i]).dot(ebeta)) / norm2[i];

			// calculate the update step
			float nextalpha = max<float>(0, min<float>(C, alpha[i] + step));
			step = nextalpha - alpha[i];
			alpha[i] = nextalpha;
			vio = max<float>(vio, abs(step));

			// update the model
			if (step != 0)
			{
				ebeta += (step * labels[i]) * eG.row(rows[i]);
				inactive_counter[i] = 0;
			}
			else if (shrinking)
			{
				inactive_counter[i]++;
				if (inactive_counter[i] >= shrink_threshold)
				{
					// shrink the point
					inactive_counter[i] = shrink_threshold - 1;
					active--;
					swap(alpha[i], alpha[active]);
					swap(labels[i], labels[active]);
					swap(inactive_counter[i], inactive_counter[active]);
					swap(rows[i], rows[active]);
					swap(norm2[i], norm2[active]);
					i--;
				}
			}
		}

		// stopping criterion
		if (vio <= epsilon)
		{
			if (shrinking && unshrink_counter != 0) unshrink_counter = 4 * n;
			else break;
		}

		unshrink_counter += active;
		if (shrinking && unshrink_counter >= 4 * n)
		{
			// declare all points as active
			active = n;
			unshrink_counter = 0;
		}
	}
}

//
// Train an SVM. The function sets the alpha matrix in the model.
//
void train(MatrixCM const& V, MatrixRM const& G, vector<float> const& norm2, vector<float> const& labels, Model& model, float C, float epsilon)
{
	WallClockTimer timer;

	uint budget = V.rows;
	uint b = V.cols;
	assert(G.rows == labels.size());
	assert(G.cols == V.cols);

	// collect binary sub-problems
	vector<pair<float, float>> problems;
	for (size_t i=0; i<model.classes.size(); i++)
	{
		for (size_t j=i+1; j<model.classes.size(); j++)
		{
			problems.push_back(make_pair(model.classes[i], model.classes[j]));
		}
	}

	EMatrixCM beta(b, problems.size());
	beta.setZero();

	// solve the binary sub-problems in parallel
	ProgressBar::start(problems.size(), "SVM training");
	#pragma omp parallel for
	for (size_t q=0; q<problems.size(); q++)
	{
		auto const& p = problems[q];

		// compose a single binary problem
		vector<uint> rows;
		vector<float> binary_norm2;
		vector<float> binary_labels;
		for (size_t k=0; k<labels.size(); k++)
		{
			if (labels[k] == p.first || labels[k] == p.second)
			{
				rows.push_back(k);
				binary_norm2.push_back(norm2[k]);
				binary_labels.push_back((labels[k] == p.first) ? -1.0f : +1.0f);
			}
		}

		// actual training
		vector<float> alpha(rows.size(), 0.0f);
		train_binary(G, binary_norm2, rows, binary_labels, alpha.data(), &beta(0, q), C, epsilon);
		ProgressBar::increment();
	}

	// compute weights from compressed weights
	model.alpha.create(budget, problems.size());
	asEigen(model.alpha) = asEigen(V) * beta;

	ProgressBar::stop();
	cout << "SVM training completed in " << timer.seconds() << " seconds" << endl;
}

//
// Train SVMs using cross-validation. The function returns the
// compressed weight matrix beta, with one block per fold and per value
// of C. It does not fill in the alpha matrix in the model.
//
MatrixCM train_cv(MatrixRM const& G, vector<float> const& norm2, vector<float> const& labels, Model const& model, uint folds, Range const& C, float epsilon, bool warmstart)
{
	WallClockTimer timer;

	uint b = G.cols;                 // compressed weight dimension

	// collect binary sub-problems
	vector<tuple<float, uint, float, float>> problems;
	for (size_t ec=0; ec<C.size(); ec++)
	{
		for (size_t f=0; f<folds; f++)
		{
			for (size_t i=0; i<model.classes.size(); i++)
			{
				for (size_t j=i+1; j<model.classes.size(); j++)
				{
					problems.push_back(make_tuple(C[ec], f, model.classes[i], model.classes[j]));
				}
			}
		}
	}

	// perform warm-starts if memory permits
	uint64_t required = 4 * (uint64_t)G.rows * (folds - 1) * C.size();
	uint64_t available = availableRAM();
	if (4 * required > 3 * available) warmstart = false;
	size_t period = problems.size() / C.size();
	vector<bool> done(problems.size(), false);
	mutex done_mutex;

	// full and compressed weight vectors
	vector<vector<float>> alpha(warmstart ? problems.size() : 0);
	MatrixCM beta(b, problems.size());

	// solve the binary sub-problems in parallel
	ProgressBar::start(problems.size(), warmstart ? "cross-validation SVM training using warm-starts" : "cross-validation SVM training using cold-starts");
	#pragma omp parallel for schedule(dynamic,1)
	for (size_t q=0; q<problems.size(); q++)
	{
		auto const& p = problems[q];

		// compose a single binary problem
		vector<uint> rows;
		vector<float> binary_norm2;
		vector<float> binary_labels;
		for (size_t k=0; k<labels.size(); k++)
		{
			if (k % folds == std::get<1>(p)) continue;
			if (labels[k] == std::get<2>(p) || labels[k] == std::get<3>(p))
			{
				rows.push_back(k);
				binary_norm2.push_back(norm2[k]);
				binary_labels.push_back((labels[k] == std::get<2>(p)) ? -1.0f : +1.0f);
			}
		}
		size_t m = rows.size();

		// prepare cold or warm start
		int origin = (int)q - (int)period;
		if (warmstart && origin >= 0)
		{
			lock_guard<mutex> lock(done_mutex);
			do
			{
				if (done[origin]) break;
				origin -= period;
			}
			while (origin >= 0);
		}
		vector<float> a(m, 0.0f);
		if (! warmstart || origin < 0)
		{
			// cold start
			for (size_t i=0; i<b; i++) beta(i, q) = 0.0f;
		}
		else
		{
			// warm start
			a = alpha[origin];
			for (size_t i=0; i<b; i++) beta(i, q) = beta(i, origin);
		}

		// actual training
		train_binary(G, binary_norm2, rows, binary_labels, a.data(), &beta(0, q), std::get<0>(p), epsilon);

		// keep the result, i.e., recover alpha based on the order of the rows
		if (warmstart)
		{
			vector<size_t> order(m);
			for (size_t i=0; i<m; i++) order[i] = i;
			std::sort(order.begin(), order.end(), [&](uint a, uint b) { return rows[a] < rows[b]; });
			alpha[q].resize(m);
			for (size_t i=0; i<m; i++) alpha[q][i] = a[order[i]];
			lock_guard<mutex> lock(done_mutex);
			done[q] = true;
		}

		ProgressBar::increment();
	}

	ProgressBar::stop();
	cout << "cross-validation SVM training completed in " << timer.seconds() << " seconds" << endl;

	return beta;
}

//
// Turn a set of binary predictions into class labels by means of voting.
//
void vote(EMatrixRM const& predictions, vector<float> const& classes, float* buffer)
{
	uint n = predictions.rows();
	uint c = classes.size();
	uint problems = c * (c-1) / 2;
	vector<uint> histogram(c);
	for (uint i=0; i<n; i++)
	{
		for (uint j=0; j<c; j++) histogram[j] = 0;
		uint a=0, b=1;
		for (uint p=0; p<problems; p++)
		{
			if (predictions(i, p) > 0) histogram[b]++; else histogram[a]++;
			b++; if (b == c) { a++; b = a+1; }
		}
		uint best_i = 0, best_n = histogram[0];
		for (uint v=1; v<histogram.size(); v++)
		{
			if (histogram[v] > best_n) { best_i = v; best_n = histogram[v]; }
		}
		buffer[i] = classes[best_i];
	}
}

//
// Compute predictions of the SVM model on data.
//
vector<float> predict(Model const& model, SparseMatrix const& data)
{
	WallClockTimer timer;

	size_t n = data.rows();
	assert(model.alpha.cols == model.classes.size() * (model.classes.size()-1) / 2);

	vector<uint> basisoffset(model.basis.offset.begin(), model.basis.offset.end());
	auto ebasis = asEigen(model.basis, basisoffset);
	auto ealpha = asEigen(model.alpha);
	vector<float> ret(n);
	size_t chunksize = 0x10000000 / model.alpha.cols;   // use at most 256 MB per thread for predictions
	if (chunksize > 4096) chunksize = 4096;
	ProgressBar::start(n, "computing predictions");
	#pragma omp parallel for
	for (size_t i=0; i<n; i+=chunksize)
	{
		uint j = min(i+chunksize, n);

		size_t z = data.offset[i];
		vector<uint> offset(j-i+1);
		for (uint k=i; k<=j; k++) offset[k-i] = data.offset[k] - z;
		auto edata = asEigen(data, offset, z);

		EMatrixRM K = kernelmatrix(edata, ebasis, model.kernel);
		EMatrixRM predictions = K * ealpha;
		vote(predictions, model.classes, ret.data() + i);

		ProgressBar::increment(j - i);
	}

	ProgressBar::stop();
	cout << "predictions completed in " << timer.seconds() << " seconds" << endl;

	return ret;
}

//
// Compute cross-validation predictions. The predictions are computed
// directly from the matrix G and the compressed weight vector. Hence,
// the function does not need to compute a kernel matrix from data.
//
vector<vector<float>> predict_cv(MatrixRM const& G, vector<float> const& classes, MatrixCM const& beta, uint folds)
{
	cout << "computing cross-validation predictions ..." << flush;
	WallClockTimer timer;

	uint n = G.rows;
	uint b = G.cols;
	uint c = classes.size();
	uint p = c * (c-1) / 2;
	uint Cs = beta.cols / (p * folds);

	vector<vector<float>> predictions(Cs, vector<float>(n));
	uint step = 0x10000000 / (p * folds);                               // use at most 256 MB per thread for predictions
	if (step > folds * (4096 / folds)) step = folds * (4096 / folds);   // 4096 rounded down to a multiple of folds
	ProgressBar::start(Cs*n, "computing cross-validation predictions");
	#pragma omp parallel for collapse(2)
	for (uint C=0; C<Cs; C++)
	{
		for (uint i=0; i<n; i+=step)
		{
			uint j = min(i+step, n);
			EMatrixRM result(j-i, p);
			for (uint f=0; f<folds; f++)
			{
				uint nn = (j-i - f + folds - 1) / folds;
				Eigen::Map<EMatrixRM const, 0, Eigen::OuterStride<>> sG(G.data() + (i+f) * b, nn, b, Eigen::OuterStride<>(folds * b));
				Eigen::Map<EMatrixCM const> sb(beta.data() + (C * folds + f) * p * b, b, p);
				Eigen::Map<EMatrixRM, 0, Eigen::OuterStride<>> sr(result.data() + f * p, nn, p, Eigen::OuterStride<>(folds * p));
				sr = sG * sb;
			}
			vote(result, classes, predictions[C].data() + i);
			ProgressBar::increment(j - i);
		}
	}

	ProgressBar::stop();
	cout << "cross-validation predictions completed in " << timer.seconds() << " seconds" << endl;

	return predictions;
}


};
