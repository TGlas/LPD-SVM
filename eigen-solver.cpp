
//
// Eigen solver using parallel processing.
//
// The only reason for this function being in its own compilation unit
// is that we don't have EIGEN_DONT_PARALLELIZE in here.
//

#include "definitions.h"

#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace std;


using EMatrixRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace cpu {

std::tuple<EMatrixRM, EMatrixRM> eig(EMatrixRM const& K)
{
	Eigen::SelfAdjointEigenSolver<EMatrixRM> solver;
	solver.compute(K, Eigen::ComputeEigenvectors);
	return std::make_tuple(solver.eigenvalues(), solver.eigenvectors());
}

};
