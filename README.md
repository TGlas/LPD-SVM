# Low-rank Parallel Dual Support Vector Machine (LPD-SVM)

This software implements highly parallel algorithms for support vector
machine training, evaluation, cross-validation, and grid-search based
parameter tuning.


## Building

A good old Makefile controls the build process. Simply run

    make -j

to build the software. For the CPU-only version, run

    make -j cpu-svm

You will probably need to change paths to your CUDA installation in the Makefile.


## License

This software was written by Tobias Glasmachers.

It is made available under the BSD-3-Clause License: https://opensource.org/licenses/BSD-3-Clause

It uses the Eigen library, which is available under the Mozilla Public License 2.0: https://www.mozilla.org/en-US/MPL/2.0/FAQ/
