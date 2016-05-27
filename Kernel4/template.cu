/*
 * Author: Shazib Hussain
 * Date: 25th April 2015
 * Organisation: Cranfield University
 *
 * This is the fourth version of the Jacobi kernel
 * This version of the kernel changes the cpu memory to page locked/pinned memory
 * This ensures that the host to device transfers are not bottlenecked by paged memory slowdowns.
 */

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
// Include CUDA files
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_timer.h>
#include <helper_cuda.h>
#include <device_functions.h>
// Defines
#define PI 3.1415926535897932384626433832795 /* Define Pi */
#define ROWS 1000	/* Number of rows */
#define COLS 1000	/* Number of columns */
#define ITER 2000	/* Number of iterations */


/* DEVICE CODE*/

// GPU Implementation of Jacobi using one cuda thread per element in the matrix
__global__ void deviceJacobi(const float *M_src, float *M_dest, int rows, int cols)
{
	// Get the theoretical 2D row and col value from the block and grid dimensions
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	// Translate this into the 1D index for the data in global memory
	int idx = ((row - 1) * rows + col);

	// Check if the value is within the boundaries
	if (row < rows && col < cols && row > 1 && col > 1) {
		// Perform the jacobi iteration
		M_dest[idx] = (0.25) * (M_src[idx + rows] + M_src[idx - rows] + M_src[idx - 1] + M_src[idx + 1]);
	}
}

/* MAIN PROGRAM*/
int main(int argc, char **argv)
{
	// CUDA Timer Setup
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

/* HOST MEMORY INITIALISATION */

	// Size of memory required
	int arrSize = ROWS * COLS * sizeof(float);

	// Allocate the matrix on the host using locked memory
	float *A;
	checkCudaErrors(cudaMallocHost((void**)&A, arrSize));
	
	//Initialise host array zeros
	for (int i = 0; i < ROWS * COLS; i++)
	{
		A[i] = 0;
	}

	// Init boundary conditions on host matrix.
	for (int i = 0; i < COLS; i++)
	{
		float t = (i + 0.0) / COLS;
		float radians = t * PI;
		float degrees = sin(radians);
		float res = pow(degrees, 2.0);
		A[i] = res;
	}

	// For Loop to get average results of time
	double counter = 0.0;
	double timeCounter = 0.0;
	int loops = 1;
	for (int z = 0; z < loops; z++) {

	/* DEVICE MEMORY INITIALISATION*/

		// Allocate GPU Memory for both matrices which are required for jacobi.
		float *A_d, *B_d;
		checkCudaErrors(cudaMalloc((void **)&A_d, arrSize));
		checkCudaErrors(cudaMalloc((void **)&B_d, arrSize));

		// Copy the host matrices to the Device
		// Time this transfer
		cudaEventRecord(start);
		checkCudaErrors(cudaMemcpy(A_d, A, arrSize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(B_d, A, arrSize, cudaMemcpyHostToDevice));
		cudaEventRecord(stop);
		cudaEventSynchronize(stop); // block cpu execution till stop recorded

		// Time to transfer data.
		float time = 0.0;
		cudaEventElapsedTime(&time, start, stop);
		timeCounter += time;


	/* DEVICE SETUP */

		// In CUDA 6.5 APIs were released for occupancy-based launches which can 
		// heuristically calculate clock sizes for maximum multiprocessor occupancy.
		// These APIs are used in this version of the kernel to achieve the maximum efficiency.
		int blockSize, minGridSize, gridSize;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, deviceJacobi, 0, (ROWS * COLS));

		// Round up according to array
		gridSize = ((ROWS*COLS) + blockSize - 1) / blockSize;

		// Turn these values into dim3 values
		double gridVal = sqrt(gridSize);
		double blockVal = sqrt(blockSize);
		gridVal = ceil(gridVal);
		blockVal = ceil(blockVal);
		int grid = (int)gridVal;
		int block = (int)blockVal;

		const dim3 BLOCK(block, block);
		const dim3 GRID(grid, grid);

	/* PERFORM COMPUTATION */

		// Perform the iterations
		// The iterations need to switch between the matrices which they update.
		// The last updated matrix will be A_d if there is an even number of iterations
		cudaEventRecord(start);
		for (int i = 0; i < ITER; i++)
		{
			// Runs Last
			if (i % 2) {
				deviceJacobi << <GRID, BLOCK >> >(B_d, A_d, ROWS, COLS);
			}
			// Runs first
			else {
				deviceJacobi << <GRID, BLOCK >> >(A_d, B_d, ROWS, COLS);
			}
		}
		cudaEventRecord(stop);

	/* CLEANUP + ANALYSIS */

		// Get execution time
		cudaEventSynchronize(stop); // block cpu execution till stop recorded
		float ms = 0.0;
		cudaEventElapsedTime(&ms, start, stop);
		//std::cout << "GPU Time: " << ms << " ms" << std::endl;

		/** Checking occupancy of the GPU **/
		/*
		cudaDeviceSynchronize();
		// Calculate Theoretical occupancy of GPU (for checking block size)
		int maxActiveBlocks, device;
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, deviceJacobi, blockSize, 0);
		cudaDeviceProp props;
		cudaGetDevice(&device);
		cudaGetDeviceProperties(&props, device);
		float occupancy = (maxActiveBlocks * blockSize / props.warpSize) /
		(float)(props.maxThreadsPerMultiProcessor / props.warpSize);
		std::cout << "Launched Blocks of size: " << blockSize << " Theoretical occupancy: " << occupancy << std::endl;
		*/

		// Get Matrix from device
		checkCudaErrors(cudaMemcpy(A, A_d, arrSize, cudaMemcpyDeviceToHost));

		// Free Memory
		checkCudaErrors(cudaFree(A_d));
		checkCudaErrors(cudaFree(B_d));

		//Add time for average
		counter += ms;
	}

	// Output averaged time
	std::cout << "Average Time: " << counter / loops << std::endl;
	std::cout << "Average Copy Time: " << timeCounter / loops << std::endl;

	// Write results to file for analysis
	// Commented out for many iterations to clculate time average.
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
	cudaFreeHost(A);

	// CUDA Cleanup
	cudaEventDestroy(stop);
	cudaEventDestroy(start);
}

