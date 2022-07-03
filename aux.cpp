
//
// General functionality (only available on the CPU)
//


#include "definitions.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <random>
#include <algorithm>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

using namespace std;


//
// Return the currently available amount of RAM in bytes.
//
uint64_t availableRAM()
{
	uint64_t pages = sysconf(_SC_AVPHYS_PAGES);
	uint64_t page_size = sysconf(_SC_PAGE_SIZE);
	return pages * page_size;
}

//
// WorkerPool implementation
//
void WorkerPool::createSingle(uint id)
{
	if (! m_threads.empty()) throwError("[WorkerPool::createSingle] The pool already contains threads.");
	m_single = id;
	m_ids.push_back(id);
}

void WorkerPool::createPool(std::vector<uint> const& ids)
{
	if (m_single >= 0) throwError("[WorkerPool::create] The pool is configured for single-threaded processing.");
	std::unique_lock<std::mutex> lock(m_mutex);
	m_ids = ids;
	for (uint id : ids)
	{
		m_individual[id] = std::list<std::function<void(uint)>>();
		m_individualLast[id] = nullptr;
		m_threads[id] = std::thread(main, id);
	}
	lock.unlock();
	while (m_running < ids.size()) std::this_thread::sleep_for(std::chrono::duration<double, std::micro>(100));
}

void WorkerPool::wait()
{
	if (m_single >= 0) return;
	while (m_incomplete > 0) std::this_thread::sleep_for(std::chrono::duration<double, std::micro>(100));
}

void WorkerPool::stop()
{
	if (m_single >= 0) return;

	m_stop = true;
	while (m_running > 0)
	{
		std::this_thread::sleep_for(std::chrono::duration<double, std::micro>(100));
		std::unique_lock<std::mutex> lock(m_mutex);
		m_condition.notify_all();
	}
	for (auto& p : m_threads) p.second.join();
}

void WorkerPool::broadcast(std::function<void(uint)> const& job)
{
	if (m_single >= 0) job(m_single);
	else
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		m_incomplete += m_individual.size();
		for (auto& p : m_individual) p.second.push_back(job);
		m_condition.notify_all();
	}
}

void WorkerPool::broadcastLast(std::function<void(uint)> const& job)
{
	if (m_single >= 0) job(m_single);
	else
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		m_incomplete += m_individual.size();
		for (auto& p : m_individualLast) p.second = job;
		m_condition.notify_all();
	}
}

void WorkerPool::enqueue(std::function<void(uint)> const& job)
{
	if (m_single >= 0) job(m_single);
	else
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		m_incomplete++;
		m_jobs.push_back(job);
		m_condition.notify_one();
	}
}

void WorkerPool::main(uint id)
{
	m_running++;
	try
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		while (true)
		{
			// wait for a trigger
			m_condition.wait(lock);

			// process all pending jobs
			while (true)
			{
				std::function<void(uint)> job;
				{
					if (! m_individual[id].empty())
					{
						job = std::move(m_individual[id].front());
						m_individual[id].pop_front();
					}
					else if (! m_jobs.empty())
					{
						job = std::move(m_jobs.front());
						m_jobs.pop_front();
					}
					else if (m_individualLast[id])
					{
						job = std::move(m_individualLast[id]);
						m_individualLast[id] = nullptr;
					}
				}
				if (job)
				{
					lock.unlock();
					job(id);
					lock.lock();
					m_incomplete--;
				}
				else break;
			}

			if (m_stop) break;
		}
	}
	catch (std::exception const& ex)
	{
		std::cout << "\n\nA critical error in GPU thread " << id << ": " << ex.what() << std::endl;
		m_stop = true;
	}
	m_running--;
}

int WorkerPool::m_single = -1;
std::vector<uint> WorkerPool::m_ids;
std::mutex WorkerPool::m_mutex;
std::condition_variable WorkerPool::m_condition;
std::atomic<uint> WorkerPool::m_running = 0;
std::atomic<uint> WorkerPool::m_incomplete = 0;
std::atomic<bool> WorkerPool::m_stop = false;
std::list<std::function<void(uint)>> WorkerPool::m_jobs;
std::unordered_map<uint, std::list<std::function<void(uint)>>> WorkerPool::m_individual;
std::unordered_map<uint, std::function<void(uint)>> WorkerPool::m_individualLast;
std::unordered_map<uint, std::thread> WorkerPool::m_threads;


//
// Return the 0/1 error between ground truth labels and predictions.
//
float error(vector<float> const& labels, vector<float> const& predictions)
{
	size_t n = labels.size();
	assert(predictions.size() == n);
	size_t e = 0;
	for (size_t i=0; i<n; i++) if (labels[i] != predictions[i]) e++;
	return (float)e / (float)n;
}

//
// Print approximation errors of the (compressed) kernel matrix.
//
void print_approximation_error(SparseMatrix const& data, Kernel const& kernel, std::vector<float> const& norm2)
{
	assert(data.rows() == norm2.size());

	// compute approximation error
	cout << "approximation error distribution:" << endl;
	uint n = min<uint>(10000, data.rows());   // estimate based on a limited sample size

	// compute exact kernel (squared feature space norm)
	vector<float> exact_norm2(data.rows());
	for (uint i=0; i<n; i++) exact_norm2[data.rows()-n+i] = kernel.norm2(data, data.rows()-n+i);

	// sort points by relative approximation error
	auto approxerror = [&norm2, &exact_norm2] (size_t i)
	{
		float q = norm2[i] / exact_norm2[i];
		return (q >= 1) ? 0.0f : 100 * sqrt(1 - q);
	};
	vector<uint> order(n);
	for (uint i=0; i<n; i++) order[i] = data.rows()-n+i;
	sort(order.begin(), order.end(), [&] (uint a, uint b) { return approxerror(a) > approxerror(b); });

	// output quantiles
	cout << "quantile   relative error\n";
	cout << "  20%:        " << approxerror(order[4 * n / 5]) << "%" << endl;
	cout << "  50%:        " << approxerror(order[n / 2]) << "%" << endl;
	cout << "  80%:        " << approxerror(order[n / 5]) << "%" << endl;
	cout << "  90%:        " << approxerror(order[n / 10]) << "%" << endl;
	cout << "  95%:        " << approxerror(order[n / 20]) << "%" << endl;
	cout << "  98%:        " << approxerror(order[n / 50]) << "%" << endl;
	cout << "  99%:        " << approxerror(order[n / 100]) << "%" << endl;
	cout << "  99.9%:      " << approxerror(order[n / 1000]) << "%" << endl;
	float e99 = approxerror(order[n / 100]);
	if (e99 > 90) cout << " The approximation error is rather large. Maybe consider using a larger budget or a broader kernel (smaller gamma).\n";
}

//
// Load a sparse data file, extract points and labels. The function
// fills in the sparse data matrix, the corresponding label vector,
// and in addition the vector classes listing the existing labels.
//
bool load(string const& filename, SparseMatrix& x, vector<float>& y, vector<float>& classes, vector<size_t>& permutation)
{
	cout << "reading data file '" << filename << "' ..." << flush;
	WallClockTimer timer;

#define F_READ(p, u, n, f) { auto r = fread(p, u, n, f); if (r != n) { fclose(file); cout << "FAILED to read file." << endl; return false; } }

	// open the file and read the "magic header"
	FILE* file = fopen(filename.c_str(), "rb");
	if (! file) { cout << "FAILED to open file." << endl; return false; }
	string magic = "....";
	F_READ(&magic[0], 1, 4, file);

	// check the file type
	if (magic == "CSR\n")
	{
		// read the binary file
		uint n;
		size_t nnz;
		F_READ(&n, sizeof(n), 1, file);
		F_READ(&x.cols, sizeof(x.cols), 1, file);
		F_READ(&nnz, sizeof(nnz), 1, file);
		y.resize(n);
		x.offset.resize(n+1);
		x.column.resize(nnz);
		x.value.resize(nnz);
		F_READ(y.data(), sizeof(float), n, file);
		F_READ(x.offset.data(), sizeof(size_t), n+1, file);
		F_READ(x.column.data(), sizeof(uint), nnz, file);
		F_READ(x.value.data(), sizeof(float), nnz, file);
		fclose(file);

		// prepare the data permutation
		permutation.resize(n);
		for (size_t i=0; i<n; i++) permutation[i] = i;
	}
	else
	{
		// assume "sparse ASCII" (LIBSVM) format
		// load the data into a string
		string content;
		if (fseek(file, 0, SEEK_END) != 0) { fclose(file); cout << "FAILED to read file." << endl; return false; }
		const std::size_t size = ftell(file);
		if (size == (std::size_t)-1) { fclose(file); cout << "FAILED to read file." << endl; return false; }
		rewind(file);
		content.resize(size);
		if (size > 0)
		{
			F_READ(&content[0], 1, size, file);
		}
		fclose(file);

		// split the file into lines
		vector<char*> lines;
		{
			char* start = &content[0];
			while (true)
			{
				while (*start == '\n') start++;
				if (*start == 0) break;
				lines.push_back(start);
				char* endline = std::strchr(start, '\n');
				*endline = 0;
				start = endline + 1;
			}
		}
		size_t n = lines.size();

		// prepare the data permutation
		permutation.resize(n);
		for (size_t i=0; i<n; i++) permutation[i] = i;

		// define a random permutation of rows
		mt19937 rng(42);
		std::shuffle(permutation.begin(), permutation.end(), rng);

		// prepare the CSR matrix
		x.offset.resize(n+1);
		x.column.clear();
		x.value.clear();
		x.cols = 0;

		// Parse the file contents, fill the matrix. This task is
		// compute-bound. It is split into blocks, which are processed in
		// parallel.
		y.resize(n);
		size_t blocks = 10 * std::thread::hardware_concurrency();
		vector<vector<uint>> column(blocks);
		vector<vector<float>> value(blocks);
		vector<uint> cols(blocks, 0);
		#pragma omp parallel
		#pragma omp for
		for (size_t b=0; b<blocks; b++)
		{
			size_t first = b * n / blocks;
			size_t last = (b+1) * n / blocks;
			for (size_t i=first; i<last; i++)
			{
				// split the line into space-separated substrings
				char* str = lines[permutation[i]];
				vector<char const*> tokens;
				while (true)
				{
					char* sep = strchr(str, ' ');
					if (sep)
					{
						*sep = 0;
						if (*str != 0) tokens.push_back(str);
						str = sep + 1;
					}
					else
					{
						if (*str != 0) tokens.push_back(str);
						break;
					}
				}

				// fill in label and features
				y[i] = strtof(tokens[0], nullptr);
				x.offset[i] = column[b].size();
				for (size_t j=1; j<tokens.size(); j++)
				{
					char* s = nullptr;
					size_t k = strtol(tokens[j], &s, 10);
					s++;
					float v = strtof(s, nullptr);
					column[b].push_back(k);
					value[b].push_back(v);
					if (k >= cols[b]) cols[b] = k + 1;
				}
			}
		}
		{
			// compose the data set from the blocks
			size_t nnz = 0;
			for (size_t b=0; b<blocks; b++)
			{
				size_t first = b * n / blocks;
				size_t last = (b+1) * n / blocks;
				for (size_t i=first; i<last; i++) x.offset[i] += nnz;
				x.cols = max(x.cols, cols[b]);
				nnz += column[b].size();
			}
			x.column.resize(nnz);
			x.value.resize(nnz);
			nnz = 0;
			for (size_t b=0; b<blocks; b++)
			{
				std::copy(column[b].begin(), column[b].end(), x.column.begin() + nnz);
				std::copy(value[b].begin(), value[b].end(), x.value.begin() + nnz);
				nnz += column[b].size();
			}
			x.offset[n] = nnz;
		}
	}

	// collect labels
	set<float> labelset;
	for (float l : y) labelset.insert(l);
	classes = vector<float>(labelset.begin(), labelset.end());

	cout << " done; " << y.size() << " points and " << classes.size() << " classes; " << timer.seconds() << " sec." << endl;

	return true;
#undef F_READ
}

//
// Save predictions to a file
//
bool save(string const& filename, vector<float> const& predictions, vector<size_t> const& permutation)
{
	size_t n = predictions.size();
	vector<float> sorted_predictions(n);
	for (size_t i=0; i<n; i++) sorted_predictions[permutation[i]] = predictions[i];
	ofstream file(filename);
	if (! file.good())
	{
		cout << "FAILED to open predictions file." << endl;
		return false;
	}
	for (size_t i=0; i<n; i++) file << sorted_predictions[i] << endl;
	return true;
}


//
// thread-safe console progress bar
//
atomic<bool> ProgressBar::m_stop = false;
atomic<bool> ProgressBar::m_active = false;
atomic<int> ProgressBar::m_maximum = 0;
atomic<int> ProgressBar::m_value = 0;
atomic<int> ProgressBar::m_prevValue = -1;
std::string ProgressBar::m_description;
std::chrono::high_resolution_clock::time_point ProgressBar::m_prevTime;
std::thread ProgressBar::m_thread;
std::mutex ProgressBar::m_mutex;

void ProgressBar::launch()
{
	lock_guard<mutex> lock(m_mutex);
	m_active = false;
	m_stop = false;
	m_thread = std::thread(threadmain);
}

void ProgressBar::shutdown()
{
	{
		lock_guard<mutex> lock(m_mutex);
		m_stop = true;
	}
	m_thread.join();
}


void ProgressBar::start(std::size_t max, std::string const& description)
{
	lock_guard<mutex> lock(m_mutex);
	m_active = true;
	m_value = 0;
	m_prevValue = -1;
	m_maximum = max;
	m_prevTime = std::chrono::high_resolution_clock::now();
	m_description = description;
}

void ProgressBar::increment(int delta)
{
	lock_guard<mutex> lock(m_mutex);
	m_value += delta;
}

void ProgressBar::stop()
{
	lock_guard<mutex> lock(m_mutex);
	m_active = false;
	cout << string(50 + m_description.size(), '\b') << string(50 + m_description.size(), ' ') << string(50 + m_description.size(), '\b');
}

void ProgressBar::threadmain()
{
	while (true)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		lock_guard<mutex> lock(m_mutex);

		if (m_stop) break;
		if (! m_active) continue;

		auto now = std::chrono::high_resolution_clock::now();
		if (m_prevValue == m_value && chrono::duration_cast<chrono::seconds>(now - m_prevTime).count() < 5) continue;

		m_prevValue = m_value.load();
		m_prevTime = now;

		cout << string(50 + m_description.size(), '\b') << string(50 + m_description.size(), ' ') << string(50 + m_description.size(), '\b');
		cout << m_description;
		int filled = (int)std::round(m_value*20.0/m_maximum);
		cout << "  ["
				<< string(filled, '#') << string(20-filled, ' ')
				<< "]  "
				<< m_value << "/" << m_maximum
				<< "  (" << 0.1*std::round(m_value*1000.0/m_maximum) << "%)";
		cout << flush;
	}
}


//
// Range methods
//
std::string Range::toString() const
{
	std::stringstream ss;
	if (low != high) ss << base << "^" << low << ":" << high;
	else ss << std::pow(base, low);
	return ss.str();
}

//
// Kernel methods
//
float Kernel::norm2(SparseMatrix const& data, uint index) const
{
	switch (type)
	{
		case Gaussian:
		{
			return 1.0f;
		}
		case Polynomial:
		{
			float sum = 0.0f;
			for (size_t i=data.offset[index]; i<data.offset[index + 1]; i++) sum += data.value[i] * data.value[i];
			return std::pow(gamma * sum + offset, (float)degree);
		}
		case TanH:
		{
			float sum = 0.0f;
			for (size_t i=data.offset[index]; i<data.offset[index + 1]; i++) sum += data.value[i] * data.value[i];
			return std::tanh(gamma * sum + offset);
		}
		default:
			throwError("unknown kernel type");
	}
}

std::string Kernel::toString(bool params) const
{
	std::stringstream ss;
	switch (type)
	{
	case Gaussian:
		ss << "Gaussian";
		if (params) ss << " gamma=" << gamma;
		break;
	case Polynomial:
		ss << "Polynomial";
		if (params) ss << " gamma=" << gamma << " degree=" << degree << " offset=" << offset;
		break;
	case TanH:
		ss << "TanH/Sigmoid";
		if (params) ss << " gamma=" << gamma << " offset=" << offset;
		break;
	default:
		throwError("unknown kernel type");
	}
	return ss.str();
}

//
// Model methods
//
bool Model::load(string const& filename)
{
#define FAIL { cout << " INVALID model file (source line " << __LINE__ << ")." << endl; return false; }
	cout << "reading model file '" << filename << "' ..." << flush;

	try
	{
		ifstream file(filename);
		if (! file.good()) { cout << " FAILED to open file." << endl; return false; }

		// read the first line
		string line;
		if (! std::getline(file, line)) FAIL
		char* str = &line[0];

		// extract the tag
		char* sep = strchr(str, ':');
		if (! sep) FAIL
		*sep = 0;
		string tag = str;
		str = sep + 1;
		while (*str == ' ') str++;

		// split the value into space-separated substrings
		vector<char const*> tokens;
		while (true)
		{
			char* sep = strchr(str, ' ');
			if (sep)
			{
				*sep = 0;
				if (*str != 0) tokens.push_back(str);
				str = sep + 1;
			}
			else
			{
				if (*str != 0) tokens.push_back(str);
				break;
			}
		}

		// parse the kernel
		if (tag != "kernel") FAIL
		{
			if (tokens.size() == 0) FAIL;
			if (tokens[0] == "Gaussian"s)
			{
				if (tokens.size() != 2) FAIL;
				kernel.type = Kernel::Gaussian;
				kernel.gamma = stof(tokens[1]);
			}
			else if (tokens[0] == "Polynomial"s)
			{
				if (tokens.size() != 4) FAIL;
				kernel.type = Kernel::Polynomial;
				kernel.gamma = stof(tokens[1]);
				kernel.degree = stof(tokens[2]);
				kernel.offset = stof(tokens[3]);
			}
			else if (tokens[0] == "TanH"s)
			{
				if (tokens.size() != 3) FAIL;
				kernel.type = Kernel::TanH;
				kernel.gamma = stof(tokens[1]);
				kernel.offset = stof(tokens[2]);
			}
			else FAIL;
		}

		// read classes as a binary vector
		size_t nc = classes.size();
		file.read((char*)(void*)&nc, sizeof(size_t));
		classes.resize(nc);
		file.read((char*)(void*)classes.data(), nc * sizeof(float));

		// read coefficients as a binary dense matrix
		file.read((char*)(void*)&alpha.rows, sizeof(alpha.rows));
		file.read((char*)(void*)&alpha.cols, sizeof(alpha.cols));
		alpha.values.resize(alpha.rows * alpha.cols);
		file.read((char*)(void*)alpha.values.data(), sizeof(float) * alpha.values.size());

		// read basis as a binary sparse matrix
		uint n = 0;
		size_t nnz = 0;
		file.read((char*)(void*)&n, sizeof(n));
		file.read((char*)(void*)&basis.cols, sizeof(uint));
		file.read((char*)(void*)&nnz, sizeof(nnz));
		basis.offset.resize(n + 1);
		basis.column.resize(nnz);
		basis.value.resize(nnz);
		file.read((char*)(void*)basis.offset.data(), sizeof(size_t) * basis.offset.size());
		file.read((char*)(void*)basis.column.data(), sizeof(uint) * basis.column.size());
		file.read((char*)(void*)basis.value.data(), sizeof(float) * basis.value.size());
	}
	catch (...)
	{
		FAIL
	}

	cout << " done." << endl;
	return true;
#undef FAIL
}

bool Model::save(string const& filename) const
{
	cout << "writing model file '" << filename << "' ..." << flush;

	ofstream file(filename, std::ios::binary | std::ios::out);
	if (! file.good())
	{
		cout << " FAILED to open file." << endl;
		return false;
	}

	// kernel header
	if (kernel.type == Kernel::Gaussian)
	{
		file << "kernel: Gaussian " << kernel.gamma << "\n";
	}
	else if (kernel.type == Kernel::Polynomial)
	{
		file << "kernel: Polynomial " << kernel.gamma << " " << kernel.degree << " " << kernel.offset << "\n";
	}
	else if (kernel.type == Kernel::TanH)
	{
		file << "kernel: TanH" << kernel.gamma << " " << kernel.offset << "\n";
	}
	else throwError("unknown kernel type");

	// classes as a binary vector
	size_t nc = classes.size();
	file.write((char const*)(void const*)&nc, sizeof(size_t));
	file.write((char const*)(void const*)classes.data(), nc * sizeof(float));

	// coefficients as a binary dense matrix
	file.write((char const*)(void const*)&alpha.rows, sizeof(alpha.rows));
	file.write((char const*)(void const*)&alpha.cols, sizeof(alpha.cols));
	file.write((char const*)(void const*)alpha.values.data(), sizeof(float) * alpha.values.size());

	// support vectors as a binary sparse matrix
	uint n = basis.rows();
	size_t nnz = basis.nnz();
	file.write((char const*)(void const*)&n, sizeof(n));
	file.write((char const*)(void const*)&basis.cols, sizeof(basis.cols));
	file.write((char const*)(void const*)&nnz, sizeof(nnz));
	file.write((char const*)(void const*)basis.offset.data(), sizeof(size_t) * basis.offset.size());
	file.write((char const*)(void const*)basis.column.data(), sizeof(uint) * basis.column.size());
	file.write((char const*)(void const*)basis.value.data(), sizeof(float) * basis.value.size());

	cout << " done." << endl;
	return true;
}
