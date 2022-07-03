
//
// Convert a sparse ASCII (LIBSVM) file into a sparse binary matrix
// (CSR) file. During conversion, the points are shuffled. This makes
// sure that the resulting data can be used for SVM training without
// further reordering.
//


#include "definitions.h"

#include <iostream>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <cstring>


using namespace std;


// Load a sparse (LIBSVM format) data file, extract points and labels.
// The function fills in the sparse data matrix and the corresponding
// label vector.
bool load(string const& filename, SparseMatrix& x, vector<float>& y)
{
	cout << "reading data file '" << filename << "' ..." << flush;

	// open the file and read the "magic code"
	FILE* file = fopen(filename.c_str(), "rb");
	if (! file) { cout << "FAILED to open file." << endl; return false; }

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
		auto r = fread(&content[0], 1, size, file);
		if (r != size) { fclose(file); cout << "FAILED to read file." << endl; return false; }
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

	// define a random permutation of rows
	vector<size_t> permutation(n);
	for (size_t i=0; i<n; i++) permutation[i] = i;
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
	#pragma omp parallel for
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

	cout << " done; "
		<< y.size() << " points."
		<< endl;

	return true;
}

void save(string const& filename, SparseMatrix const& x, vector<float> const& y)
{
	cout << "writing data file '" << filename << "' ..." << flush;

	FILE* file = fopen(filename.c_str(), "wb+");
	if (! file) { cout << "FAILED to open file." << endl; }
	string magic = "CSR\n";
	fwrite(magic.data(), 1, 4, file);

	uint n = y.size();
	size_t nnz = x.value.size();
	assert(x.offset.size() == n+1);
	assert(x.column.size() == nnz);

	fwrite(&n, sizeof(n), 1, file);
	fwrite(&x.cols, sizeof(x.cols), 1, file);
	fwrite(&nnz, sizeof(nnz), 1, file);
	fwrite(y.data(), sizeof(float), n, file);
	fwrite(x.offset.data(), sizeof(size_t), n+1, file);
	fwrite(x.column.data(), sizeof(uint), nnz, file);
	fwrite(x.value.data(), sizeof(float), nnz, file);
	fclose(file);

	cout << " done." << endl;
}


int main(int argc, char** argv)
{
	// parse command line parameters
	if (argc != 3)
	{
		cout << "usage: " << argv[0] << " <inputfile> <outputfile>\n"
		        "with:\n"
		        "  <inputfile>    sparse ASCII file in LIBSVM format\n"
		        "  <outputfile>   binary CSR file\n";
		return EXIT_FAILURE;
	}

	// load the data from disk
	SparseMatrix x;
	vector<float> y, classes;
	if (! load(argv[1], x, y)) return EXIT_FAILURE;

	// save it in binary format
	save(argv[2], x, y);

	return EXIT_SUCCESS;
}
