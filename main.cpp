
//
// Main program with command line interface and distribution of the
// given tasks to CPU and GPU.
//

#include "definitions.h"
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;


#ifdef DEBUG
// enable this macro to let exceptions pass through to the debugger
#define NO_TRY_CATCH
#endif


namespace cpu {
	void analyze(Model const& model, SparseMatrix const& data, vector<float> const& labels);
};


//
// Display the help text for the CLI.
//
void help(string const& exec)
{
	cout << "\n"
#ifdef WITH_GPU_SUPPORT
			"Low-Rank Parallel Dual SVM training on CPU and GPU.\n"
#else
			"Low-Rank Parallel Dual SVM training on the CPU.\n"
#endif
			"Copyright (c) 2021-2022 Tobias Glasmachers\n"
			"This software is made available under the BSD-3-Clause License:\n"
			"https://opensource.org/licenses/BSD-3-Clause\n"
			"\n"
			"The software provides the following modes of operation:\n"
			" - Train an SVM model\n"
			"     " << exec << " train <datafile> <modelfile> <C> <kernel> <options>\n"
			" - Compute predictions of a model on data (ignoring the labels)\n"
			"     " << exec << " predict <datafile> <modelfile> <options> <predictions>\n"
			" - Compute the error of the data on the model\n"
			"     " << exec << " test <datafile> <modelfile> <options>\n"
			" - Compute the cross-validation error of a training configuration\n"
			"     " << exec << " cross-validate <datafile> <folds> <C> <kernel> <options>\n"
			"\n"
			"The data must be in sparse (LIBSVM) data format, or in our own binary CSR\n"
			"format. The convert-to-csr tool can be used for conversion. Loading data in\n"
			"binary format is significantly faster.\n"
			"\n"
			"The parameter C is a positiv number. In cross-validation mode, is can be a\n"
			"range: It can be specified as a (decimal) number, or as a power b^n, where n is\n"
			"an integer. A range is specified as b^n:m, where n and m are integers, and n<m.\n"
			"It represents the set of values {b^k | n <= k <= m}. Example: 2^-3:7\n"
			"\n"
			"Three types of kernels are supported, with the following parameters:\n"
			"  KERNEL TYPE       PARAMETERS               FORMULA\n"
			"  Gaussian          gamma                    exp(-gamma*dist(x,y)^2)\n"
			"  Polynomial        gamma, degree, offset    (gamma*inner(x,y)+offset)^degree\n"
			"  TanH (sigmoid)    gamma, offset            tanh(gamma*inner(x,y)+offset)\n"
			"Here, dist is the Euclidean distance and inner is the inner product between\n"
			"two data points. The kernel is specified by its (case-insensitive) name.\n"
			"Parameters are set to default values as follows.\n"
			"  PARAMETER          DEFAULT\n"
			"  gamma              1 / (estimated) squared norm of a typical input vector\n"
			"  offset             0\n"
			"  degree             3\n"
			"The defaults can be overridden using options. In cross-validation mode, gamma\n"
			"can be specified as a range. For gamma and degree, it is recommended to tune\n"
			"instead of relying on the defaults.\n"
			"\n"
			"The following options are available:\n"
			" -gamma <range>      kernel parameter, see above\n"
			" -degree <number>    kernel parameter, see above\n"
			" -offset <number>    kernel parameter, see above\n"
			" -epsilon <number>   target precision epsilon, default=0.01\n"
			" -budget <integer>   number of basis points, default=1000\n"
			" -coldstart          disable warm-starts in cross-validation mode\n"
			" -approx             display the approximation error distribution\n"
#ifdef WITH_GPU_SUPPORT
			" -gpu <numbers>      select GPUs (comma-separated list of CUDA device IDs\n"
			"                     or 'all'); default=all\n"
			" -prepare <device>   device is 'cpu' or 'gpu'; default=gpu\n"
			" -train <device>     device is 'cpu' or 'gpu'; default=cpu\n"
			" -predict <device>   device is 'cpu' or 'gpu'; default=gpu\n"
			"The last three options control where to perform the preparation of the kernel\n"
			"matrix, the actual training (SMO loop), and the prediction.\n"
#endif
			"\n";
	exit(EXIT_FAILURE);
}

//
// entry point
//
int main(int argc, char** argv)
{
	// parse command line parameters
	auto parseRange = [argv] (string const& s)
	{
		Range r;
		try
		{
			size_t power = s.find('^');
			if (power != string::npos)
			{
				r.base = stof(s.substr(0, power));
				size_t colon = s.find(':');
				if (colon != string::npos)
				{
					r.low = stoul(s.substr(power+1, colon-power-1));
					r.high = stoul(s.substr(colon+1));
				}
				else r.low = r.high = stoul(s.substr(power+1));
			}
			else
			{
				r.base = stof(s);
				r.low = 1;
				r.high = 1;
			}
			if (r.base <= 0) help(argv[0]);
			if (r.low > r.high) help(argv[0]);
		}
		catch (...)
		{
			help(argv[0]);
		}
		return r;
	};

	if (argc < 3) help(argv[0]);
	string mode = argv[1];
	if (mode != "train" && mode != "predict" && mode != "test" && mode != "cross-validate") help(argv[0]);
	string datafile = argv[2];
	string modelfile;
	string predictionfile;
	Range C, kernelgamma;
	Model model;
	float epsilon = 0.01f;
	uint budget = 1000, folds = 0;
	bool printApproxError = false;
	bool warmstart = true;
#ifdef WITH_GPU_SUPPORT
	vector<uint> selectedDevices;
	bool prepareGPU = true;
	bool trainGPU = false;
	bool predictGPU = true;
#endif
	uint optionstart = (mode == "predict") ? 5 : ((mode == "test") ? 4 : 6);
	if ((uint)argc < optionstart) help(argv[0]);

	try
	{
		if (mode == "cross-validate")
		{
			folds = stoul(argv[3]);
			if (folds < 2) help(argv[0]);
		}
		else modelfile = argv[3];
		if (mode == "predict")
		{
			predictionfile = argv[4];
		}
		if (mode == "train" || mode == "cross-validate")
		{
			C = parseRange(argv[4]);
			string arg = argv[5];
			for (size_t i=0; i<arg.size(); i++) if (arg[i] >= 'A' && arg[i] <= 'Z') arg[i] += 32;
			if (arg == "gaussian") model.kernel.type = Kernel::Gaussian;
			else if (arg == "polynomial") model.kernel.type = Kernel::Polynomial;
			else if (arg == "tanh") model.kernel.type = Kernel::TanH;
			else help(argv[0]);
		}

		for (int a=optionstart; a<argc; a++)
		{
			string arg = argv[a];
			if (arg == "-gamma")
			{
				a++; if (a >= argc) help(argv[0]);
				kernelgamma = parseRange(argv[a]);
			}
			else if (arg == "-degree")
			{
				a++; if (a >= argc) help(argv[0]);
				model.kernel.degree = stof(argv[a]);
				if (model.kernel.degree <= 0) help(argv[0]);
			}
			else if (arg == "-offset")
			{
				a++; if (a >= argc) help(argv[0]);
				model.kernel.offset = stof(argv[a]);
			}
			else if (arg == "-epsilon")
			{
				a++; if (a >= argc) help(argv[0]);
				epsilon = stof(argv[a]);
				if (epsilon <= 0) help(argv[0]);
			}
			else if (arg == "-budget")
			{
				a++; if (a >= argc) help(argv[0]);
				budget = stoul(argv[a]);
				if (budget <= 0) help(argv[0]);
			}
			else if (arg == "-approx")
			{
				printApproxError = true;
			}
			else if (arg == "-coldstart")
			{
				warmstart = false;
			}
#ifdef WITH_GPU_SUPPORT
			else if (arg == "-gpu")
			{
				a++; if (a >= argc) help(argv[0]);
				arg = argv[a];
				if (arg != "all")
				{
					// comma-separated list
					size_t start = 0;
					while (true)
					{
						std::size_t pos = arg.find(',', start);
						uint d = stoul(arg.substr(start, pos-start));
						selectedDevices.push_back(d);
						if (pos == string::npos) break;
						start = pos + 1;
					}
				}
			}
			else if (arg == "-prepare")
			{
				a++; if (a >= argc) help(argv[0]);
				arg = argv[a];
				if (arg != "cpu" && arg != "gpu") help(argv[0]);
				prepareGPU = (arg == "gpu");
			}
			else if (arg == "-train")
			{
				a++; if (a >= argc) help(argv[0]);
				arg = argv[a];
				if (arg != "cpu" && arg != "gpu") help(argv[0]);
				trainGPU = (arg == "gpu");
			}
			else if (arg == "-predict")
			{
				a++; if (a >= argc) help(argv[0]);
				arg = argv[a];
				if (arg != "cpu" && arg != "gpu") help(argv[0]);
				predictGPU = (arg == "gpu");
			}
#endif
			else
			{
				cout << "invalid command line argument: " << arg << endl;
				help(argv[0]);
			}
		}
		if (mode == "train" && (C.low != C.high || kernelgamma.low != kernelgamma.high)) help(argv[0]);
	}
	catch (...)
	{
		help(argv[0]);
	}

	// load the data from disk
	SparseMatrix x;
	vector<float> y, classes;
	vector<size_t> permutation;
	if (! load(datafile, x, y, classes, permutation)) return EXIT_FAILURE;

	ProgressBar::launch();

#ifdef WITH_GPU_SUPPORT
	{
		// detect CUDA-capable GPUs
		gpu::detectCudaDevices();
		if (cudaDevices.empty())
		{
			cout << "\n\nNo CUDA-capable GPUs found - exiting." << endl;
			ProgressBar::shutdown();
			return EXIT_FAILURE;
		}
		if (selectedDevices.empty())
		{
			if ((mode == "train" && x.rows() * classes.size() < 200000)
				|| ((mode == "test" || mode == "predict") && x.rows() * classes.size() * (classes.size()-1) / 2 < 10000000))
			{
				// heuristic: small task, use only the strongest GPU
				vector<uint> candidates;
				for (auto const& p : cudaDevices) candidates.push_back(p.first);
				selectedDevices.push_back(gpu::selectMainCudaDevice(candidates));
			}
			else
			{
				// use all available GPUs
				for (auto const& p : cudaDevices) selectedDevices.push_back(p.first);
			}
		}
		else
		{
			for (uint id : selectedDevices)
			{
				if (cudaDevices.count(id) == 0)
				{
					cout << "\n\nGPU " << id << " is not a valid CUDA device - exiting." << endl;
					ProgressBar::shutdown();
					return EXIT_FAILURE;
				}
			}
		}
		if (selectedDevices.size() == 1) WorkerPool::createSingle(selectedDevices[0]);
		else WorkerPool::createPool(selectedDevices);
	}
#endif

	auto cleanup = [&]()
	{
		ProgressBar::shutdown();
#ifdef WITH_GPU_SUPPORT
		WorkerPool::broadcast([](uint id){ gpu::quit(id); });
		WorkerPool::stop();
#endif
	};

	// if unspecified, set the kernel scale/width to its default
	if (kernelgamma.base == 0)
	{
		// inverse of the median squared norm of a subset of the data
		size_t n = 999;
		if (x.rows() < n) n = x.rows();
		if ((n & 1) == 0) n--;
		std::vector<float> norm2(n);
		for (size_t i=0; i<n; i++)
		{
			float sum = 0;
			size_t o = x.offset[i];
			size_t u = x.offset[i+1];
			for (size_t j=o; j<u; j++) sum += x.value[j] * x.value[j];
			norm2[i] = sum;
		}
		std::nth_element(norm2.begin(), norm2.begin() + n/2, norm2.end());
		kernelgamma.base = 1.0 / norm2[n/2];
		kernelgamma.low = 1;
		kernelgamma.high = 1;
	}

	// if too large, correct the budget size
	if (budget > x.rows()) budget = x.rows();

	// print out a summary of the configuration
	cout << endl;
	cout << "  mode:                      " << mode << endl;
	cout << "  dataset:                   " << datafile << endl;
	if (mode != "cross-validate") cout << "  model file:                " << modelfile << endl;
	else                          cout << "  number of folds:           " << folds << endl;
	if (mode == "train" || mode == "cross-validate")
	{
		cout << "  C:                         " << C.toString() << endl;
		cout << "  kernel:                    " << model.kernel.toString(false) << endl;
		cout << "   gamma:                     " << kernelgamma.toString() << endl;
		if (model.kernel.type == Kernel::Polynomial) cout << "   degree:                    " << model.kernel.degree << endl;
		if (model.kernel.type != Kernel::Gaussian)   cout << "   offset:                    " << model.kernel.offset << endl;
		cout << "  target precision:          " << epsilon << endl;
		cout << "  budget size:               " << budget << endl;
	}
	if (mode == "cross-validate")
	{
		cout << "  warm-starts:               " << (warmstart ? "enabled" : "disabled") << endl;
	}
	if (mode == "predict") cout << "  prediction file:           " << modelfile << endl;
#ifdef WITH_GPU_SUPPORT
	cout << "  GPUs (CUDA devices):       ";
	for (size_t i=0; i<selectedDevices.size(); i++)
	{
		if (i > 0) cout << ",";
		cout << selectedDevices[i];
	}
	cout << endl;
	if (mode == "train" || mode == "cross-validate")
	{
		cout << "  kernel matrix preparation: " << (prepareGPU ? "gpu" : "cpu") << endl;
		cout << "  training:                  " << (trainGPU ? "gpu" : "cpu") << endl;
	}
	if (mode != "train")
	{
		cout << "  prediction:                " << (predictGPU ? "gpu" : "cpu") << endl;
	}
#endif
	cout << endl;

#ifdef WITH_GPU_SUPPORT
	// Initialize CUDA. This step can take up to several seconds per GPU.
	{
		WallClockTimer timer;
		ProgressBar::start(selectedDevices.size(), "initializing CUDA");
		WorkerPool::broadcast([](uint id)
			{
				gpu::init(id);
				cuda_sync();
				ProgressBar::increment();
			});
		WorkerPool::wait();
		ProgressBar::stop();
		cout << "CUDA initialized in " << timer.seconds() << " seconds" << endl;
	}
#endif

	// Create or load a set of basis vectors. Used in "train" and "cross-validation" modes.
	auto prepareBasis = [&]()
	{
		WallClockTimer timer;
		cout << "preparing basis ..." << flush;
		{
			// create the basis

			// use the first chunk (in effect, a random subset of
			// the shuffled data) as the basis (support vectors)
			size_t nnz = x.offset[budget];
			model.basis.offset.resize(budget + 1);
			model.basis.column.resize(nnz);
			model.basis.value.resize(nnz);
			model.basis.cols = x.cols;

			// sort the basis points by the number of non-zeros in decreasing order
			vector<size_t> order(budget);
			for (size_t i=0; i<budget; i++) order[i] = i;
			std::sort(order.begin(), order.end(), [&x](size_t a, size_t b) { return x.nnz(a) > x.nnz(b); });

			// copy the column/value data
			size_t start = 0;
			for (uint i=0; i<budget; i++)
			{
				size_t j = order[i];
				size_t s = x.offset[j];
				size_t e = x.offset[j+1];
				model.basis.offset[i] = start;
				std::copy(&x.column[s], &x.column[e], &model.basis.column[start]);
				std::copy(&x.value[s], &x.value[e], &model.basis.value[start]);
				start += e - s;
			}
			model.basis.offset[budget] = start;
			assert(start == nnz);
		}

#ifdef WITH_GPU_SUPPORT
		for (uint id : WorkerPool::workerIDs()) model.gpu_basis[id] = GpuSparseMatrix();
		WorkerPool::broadcast([&](uint id){ model.gpu_basis[id] = GpuSparseMatrix(model.basis); });
#endif
		cout << " done; completed in " << timer.seconds() << " seconds" << endl;
	};

	// mode-dependent processing
	if (mode == "train")
	{
#ifndef NO_TRY_CATCH
		try
#endif
		{
			prepareBasis();

			model.classes = classes;
			model.kernel.gamma = kernelgamma[0];

#ifdef WITH_GPU_SUPPORT
			for (uint id : WorkerPool::workerIDs()) model.gpu_classes[id] = DeviceArray<float>();
			WorkerPool::broadcast([&](uint id){ model.gpu_classes[id].create(classes); });

			if (prepareGPU && trainGPU && WorkerPool::size() == 1)
			{
				// keep all data on the same GPU, i.e., avoid memory transfers in between prepare and train
				WorkerPool::enqueue(
					[&](uint id)
					{
						uint b = 0;
						DeviceArray<float> V, G, norm2;
						tie(b, V, G, norm2) = gpu::prepare_for_gpu(id, x, model.gpu_basis[id], model.kernel, budget);
						if (printApproxError)
						{
							vector<float> cpu_norm2(norm2.size());
							norm2.to_cpu(cpu_norm2);
							print_approximation_error(x, model.kernel, cpu_norm2);
						}
						gpu::train(id, V, G, norm2, y, model, C[0], epsilon);
					}
				);
			}
			else
			{
				// general case: possibly multi-GPU, or even CPU
#endif

				uint b = 0;
				MatrixCM V;
				MatrixRM G;
				vector<float> norm2;

#ifdef WITH_GPU_SUPPORT
				WorkerPool::wait();
				if (prepareGPU) tie(b, V, G, norm2) = gpu::prepare_for_cpu(x, model.gpu_basis, model.kernel, budget);
				else
#endif
				tie(b, V, G, norm2) = cpu::prepare(x, model.basis, model.kernel, budget);

				if (printApproxError) print_approximation_error(x, model.kernel, norm2);

#ifdef WITH_GPU_SUPPORT
				if (trainGPU) gpu::train(V, G, norm2, y, model, C[0], epsilon);
				else cpu::train(V, G, norm2, y, model, C[0], epsilon);
			}
#else
			cpu::train(V, G, norm2, y, model, C[0], epsilon);
#endif

			{
				WallClockTimer timer;
				ProgressBar::start(1, "saving model file");
				if (! model.save(modelfile)) { cleanup(); return EXIT_FAILURE; }
				ProgressBar::increment();
				ProgressBar::stop();
				cout << "model file saved in " << timer.seconds() << " seconds" << endl;
			}
		}
#ifndef NO_TRY_CATCH
		catch (std::exception const& ex)
		{
			cout << "\n\nA critical error occurred: " << ex.what() << endl;
		}
#endif
	}
	else if (mode == "predict")
	{
#ifndef NO_TRY_CATCH
		try
#endif
		{
			std::vector<float> predictions;
#ifdef WITH_GPU_SUPPORT
			if (predictGPU)
			{
				if (! model.load(modelfile)) { cleanup(); return EXIT_FAILURE; }

				for (uint id : WorkerPool::workerIDs())
				{
					model.gpu_classes[id] = DeviceArray<float>();
					model.gpu_basis[id] = GpuSparseMatrix();
					model.gpu_alpha[id] = DeviceArray<float>();
				}
				WorkerPool::broadcast([&](uint id){
						model.gpu_classes[id].create(classes);
						model.gpu_basis[id] = GpuSparseMatrix(model.basis);
						model.gpu_alpha[id].create(model.alpha.rows * model.alpha.cols, model.alpha.data());
					});

				// correct the data dimension
				x.cols = model.basis.cols = max(x.cols, model.basis.cols);

				predictions = gpu::predict(model, x);
			}
			else
#endif
			{
				if (! model.load(modelfile)) { cleanup(); return EXIT_FAILURE; }

				// correct the data dimension
				x.cols = model.basis.cols = max(x.cols, model.basis.cols);

				predictions = cpu::predict(model, x);
			}

			// store the predictions to a file
			if (! save(predictionfile, predictions, permutation)) { cleanup(); return EXIT_FAILURE; }
		}
#ifndef NO_TRY_CATCH
		catch (std::exception const& ex)
		{
			cout << "\n\nA critical error occurred: " << ex.what() << endl;
		}
#endif
	}
	else if (mode == "test")
	{
#ifndef NO_TRY_CATCH
		try
#endif
		{
			std::vector<float> predictions;
#ifdef WITH_GPU_SUPPORT
			if (predictGPU)
			{
				if (! model.load(modelfile)) { cleanup(); return EXIT_FAILURE; }

				for (uint id : WorkerPool::workerIDs())
				{
					model.gpu_classes[id] = DeviceArray<float>();
					model.gpu_basis[id] = GpuSparseMatrix();
					model.gpu_alpha[id] = DeviceArray<float>();
				}
				WorkerPool::broadcast([&](uint id){
						model.gpu_classes[id].create(classes);
						model.gpu_basis[id] = GpuSparseMatrix(model.basis);
						model.gpu_alpha[id].create(model.alpha.rows * model.alpha.cols, model.alpha.data());
					});

				// correct the data dimension
				x.cols = model.basis.cols = max(x.cols, model.basis.cols);

				predictions = gpu::predict(model, x);
			}
			else
#endif
			{
				if (! model.load(modelfile)) { cleanup(); return EXIT_FAILURE; }

				// correct the data dimension
				x.cols = model.basis.cols = max(x.cols, model.basis.cols);

				predictions = cpu::predict(model, x);
			}
			float e = error(y, predictions);
			cout << "***   error rate: " << (100 * e) << "%   ***" << endl;
		}
#ifndef NO_TRY_CATCH
		catch (std::exception const& ex)
		{
			cout << "\n\nA critical error occurred: " << ex.what() << endl;
		}
#endif
	}
	else if (mode == "cross-validate")
	{
#ifndef NO_TRY_CATCH
		try
#endif
		{
			prepareBasis();

			model.classes = classes;
#ifdef WITH_GPU_SUPPORT
			for (uint id : WorkerPool::workerIDs()) model.gpu_classes[id] = DeviceArray<float>();
			WorkerPool::broadcast([&](uint id){ model.gpu_classes[id].create(classes); });
#endif

			// grid of cross-validation errors
			vector<float> cv_error;

			for (int eg=kernelgamma.low; eg<=kernelgamma.high; eg++)
			{
				model.kernel.gamma = std::pow(kernelgamma.base, eg);

				vector<vector<float>> predictions;   // one vector of predictions for each value of C

				uint b = 0;
				MatrixCM V;
				MatrixRM G;
				vector<float> norm2;

				// prepare the kernel matrix
#ifdef WITH_GPU_SUPPORT
				WorkerPool::wait();
				if (prepareGPU) tie(b, V, G, norm2) = gpu::prepare_for_cpu(x, model.gpu_basis, model.kernel, budget);
				else
#endif
				tie(b, V, G, norm2) = cpu::prepare(x, model.basis, model.kernel, budget);

				// approximation error
				if (printApproxError) print_approximation_error(x, model.kernel, norm2);

				// cross-validation training
				MatrixCM beta;
#ifdef WITH_GPU_SUPPORT
				unordered_map<uint, DeviceArray<float>> gpu_G;
				if (trainGPU)
				{
					for (uint id : WorkerPool::workerIDs()) gpu_G[id] = DeviceArray<float>();
					WorkerPool::broadcast([&](uint id)
						{
							gpu_G[id].create(G.values.size(), G.data());
						});
				}
				WorkerPool::wait();
				if (trainGPU) beta = gpu::train_cv(gpu_G, norm2, y, model, folds, C, epsilon, warmstart);
				else
#endif
				beta = cpu::train_cv(G, norm2, y, model, folds, C, epsilon, warmstart);

				// cross-validation prediction
#ifdef WITH_GPU_SUPPORT
				if (predictGPU)
				{
					if (trainGPU)
						predictions = gpu::predict_cv(gpu_G, model.gpu_classes, beta, folds);
					else
						predictions = gpu::predict_cv(G, model.gpu_classes, beta, folds);
				}
				else
#endif
				predictions = cpu::predict_cv(G, model.classes, beta, folds);

				// cross-validation error rates
				for (auto const& p : predictions)
				{
					cv_error.push_back(error(y, p));
				}
			}

			// final report on error rates over the parameter grid
			cout << "cross validation error rates:" << endl;
			float minimum = *std::min_element(cv_error.begin(), cv_error.end());
			size_t j = 0;
			for (size_t eg=0; eg<kernelgamma.size(); eg++)
			{
				for (size_t ec=0; ec<C.size(); ec++)
				{
					if (cv_error[j] == minimum) cout << "***\t"; else cout << "   \t";
					cout << "gamma = " << kernelgamma[eg] << "\t";
					cout << "    C = " <<           C[ec] << "\t";
					cout << "    cross-validation error = " << (100 * cv_error[j]) << "%" << endl;
					j++;
				}
			}
		}
#ifndef NO_TRY_CATCH
		catch (std::exception const& ex)
		{
			cout << "\n\nA critical error occurred: " << ex.what() << endl;
		}
#endif
	}

	cleanup();
	return EXIT_SUCCESS;
}
