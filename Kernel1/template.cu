/*
 * Author: Shazib Hussain
 * Date: 25th April 2015
 * Organisation: Cranfield University
 *
 * This is the first version of the CUDA Jacobi Kernel
 * No attempt at any kind of optimising behaviour is made
 * This is the most basic kernel and it performs its computation on global memory
 */

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
// Include CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_timer.h>
#include <helper_cuda.h>
// Defines
#define PI 3.1415926535897932384626433832795 /* Define Pi */
#define ROWS 1000	/* Number of rows */
#define COLS 1000	/* Number of columns */
#define ITER 2000	/* Number of iterations */


/* DEVICE CODE*/

// GPU Implementation of Jacobi using one cuda thread per element in the matrix
// Pointers to the new and the old matrix are passed to the kernel as well as the num of rows + cols
__global__ void deviceJacobi(const double *M_src, double *M_dest, int rows, int cols)
{
	// Get the theoretical 2D row and col value from the block and grid dimensions
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	// Translate this into the 1D index for the data in memory
	int idx = ((row-1) * rows + col);

	// Check if the value is within the boundaries
	if (row < rows && col < cols && row > 1 && col > 1) {
		// Perform the jacobi iteration
		M_dest[idx] =  (0.25) * (M_src[idx+rows] + M_src[idx-rows] + M_src[idx-1] + M_src[idx+1]);
	}
}

/* MAIN PROGRAM*/
int main(int argc, char **argv)
{
	// CUDA Timer Setup
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Size of memory needed
	int arrSize = ROWS * COLS * sizeof(double);

/* HOST MEMORY INITIALISATION */

	// Allocate the matrix on the host
	double *A = new double[ROWS * COLS];

	//Initialise to zeros
	for (int i = 0; i < ROWS * COLS; i++)
	{
		A[i] = 0;
	}

	// Init boundary conditions on only A matrix.
	for (int i = 0; i < COLS; i++)
	{
		double t = (i + 0.0) / COLS;
		double radians = t * PI;
		double degrees = sin(radians);
		double res = pow(degrees, 2.0);
		A[i] = res;
	}

	// For Loop to get average results of time
	float counter = 0.0;
	int loops = 100;
	for (int z = 0; z < loops; z++) {

	/* DEVICE MEMORY INITIALISATION*/

		// Allocate GPU Memory for both matrices which are required for jacobi.
		double *A_d, *B_d;
		checkCudaErrors(cudaMalloc((void **)&A_d, arrSize));
		checkCudaErrors(cudaMalloc((void **)&B_d, arrSize));

		// Copy matrices to Device
		checkCudaErrors(cudaMemcpy(A_d, A, arrSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(B_d, A, arrSize, cudaMemcpyHostToDevice));



	/* DEVICE SETUP */

		// Computer grid and block size
		// The maximum allowed threads per block for compute capability 2.1 is 1024.
		// The required number of threads for our 100 X 100 matrix is 10000, Excluding boundary values 9801.
		// Therefore a minimum of 10 blocks is required.
		// Some more complex arthitmetic can be used to calculate the number of required blocks.
		const dim3 BLOCK_DIM(32, 32); // 1024 threads
		//const dim3 GRID_SIZE(30); // 1024 * 20, minimum required
		const dim3 GRID_DIM((COLS - 1) / BLOCK_DIM.x + 1, (ROWS - 1) / BLOCK_DIM.y + 1);

	/* PERFORM COMPUTATION */

		// Perform the iterations
		// The iterations need to switch between the matrices which they update.
		// The last updated matrix will be A_d if there is an even number of iterations
		cudaEventRecord(start, 0);
		for (int i = 0; i < ITER; i++)
		{
			// Runs Last
			if (i % 2) {
				deviceJacobi <<<GRID_DIM, BLOCK_DIM >>>(B_d, A_d, ROWS, COLS);
			}
			// Runs first
			else {
				deviceJacobi <<<GRID_DIM, BLOCK_DIM >>>(A_d, B_d, ROWS, COLS);
			}
		}
		cudaEventRecord(stop, 0);

	/* CLEANUP + ANALYSIS */

		// Get time of execution
		cudaEventSynchronize(stop); // block cpu execution till stop recorded
		float ms = 0.0;
		cudaEventElapsedTime(&ms, start, stop);
		//std::cout << "GPU Time: " << ms << " ms" << std::endl;

		// Get Matrix from device
		checkCudaErrors(cudaMemcpy(A, A_d, arrSize, cudaMemcpyDeviceToHost));

		// Free Memory
		checkCudaErrors(cudaFree(A_d));
		checkCudaErrors(cudaFree(B_d));

		// Add time for average
		counter += ms;
	}
	// Output averaged time
	std::cout << "Average GPU Time: " << counter / loops << std::endl;

	// Write results to file for analysis
	FILE *fp;
	fopen_s(&fp, "jacobi.dat", "w");
	int idx = 0;
	for (int i = 0; i < ROWS * COLS; i++)
	{
		idx++;
		fprintf(fp, "%.6f ", A[i]);

		if (idx == COLS) {
			idx = 0;
			fprintf(fp, "\n");
		}
	}
	fclose(fp);

	// Clear Host Memory
	delete(A);

	// CUDA Cleanup
	cudaEventDestroy(stop);
	cudaEventDestroy(start);
}
