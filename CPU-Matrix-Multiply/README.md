# OPTIMISING MATRIX MULTIPLY WITH CACHE BLOCKING AND AVX2 MICROKERNEL

Description
===========
In this project, we use our knowledge of the memory hierarchy and vectorization to optimize a matrix multiply routine. We compare it with BLAS, which is an optimized basic linear algebra subroutine. We optimize our code for maximum performance on one core of an Amazon EC2 t2.micro instance.

Code organization
=================
* blislab/bl_dgemm_ukr.c - implements AVX2 microkernel
* blislab/my_dgemm.c - run this to perform blocked matrix multiply. Use makefile to compile.
* blas - contains an BLAS matrix multiply routine
* naive - contains an naive matrix multiply routine

Results
=======
<img src="performance.png" width="600">

Acknowledgements
================
I thank Prof. Bryan Chin and TAs for their guidance and support.

