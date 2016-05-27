/*
* Author: Shazib Hussain
* Date: 25th April 2015
* Organisation: Cranfield University
*
* This is the fifth version of the Jacobi kernel
* This version of the kernel addresses memory issues by using cudaMallocPitch
* and cudaMemcpy2D to pad the memory out for contingiious access.
*/

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
// Include CUDA
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
__global__ void deviceJacobi(const float *M_src, float *M_dest, int rows, int cols, size_t pitch)
{

	// Get the theoretical 2D row and col value from the block and grid dimensions
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	// If in bounds
	if (row < 1000 && row > 0 && col < 1000 && col > 0) {
		// Get rows
		float* rowDest = (float*)(((char*)M_dest) + ((row)* pitch));
		float* rowUp = (float*)(((char*)M_src) + ((row-1)* pitch));
		float* rowDown = (float*)(((char*)M_src) + ((row+1)* pitch));
		float* rowSrc = (float*)(((char*)M_src) + ((row)* pitch));

		// Move to shared
		extern __shared__ float rowU[];
		extern __shared__ float rowD[];
		extern __shared__ float rowS[]; 
		rowU[col] = rowUp[col];
		rowD[col] = rowDown[col];
		rowS[col] = rowSrc[col];
		__syncthreads();

		// Perform jacobi
		rowDest[col] = (0.25) *  (rowS[col - 1] + rowS[col + 1] + rowD[col] + rowU[col]);
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

	// Allocate the matrix on the host
	float *A;

	/checkCudaErrors(cudaMallocHost((float**)&A, ROWS*COLS*sizeof(float)));

	//Initialise to zeros
	for (int i = 0; i < ROWS*COLS; i++)
	{
		A[i] = 0;
	}

	// Init boundary conditions on only A matrix.
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

		// Allocate padded memory in the GPU
		// We do not pad the host array, as this will then possibly require more padding on the device
		size_t hostPitchA = COLS * sizeof(float);
		size_t hostPitchB = COLS * sizeof(float);
		size_t devPitchA, devPitchB;

		checkCudaErrors(cudaMallocPitch(&A_d, &devPitchA, COLS*sizeof(float), ROWS));
		checkCudaErrors(cudaMallocPitch(&B_d, &devPitchB, COLS*sizeof(float), ROWS));
		
		// Copy the matrices to the Device
		// Time this transfer
		cudaEventRecord(start);
		checkCudaErrors(cudaMemcpy2D(A_d, devPitchA, A, hostPitchA, sizeof(float)*COLS, ROWS, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy2D(B_d, devPitchB, A, hostPitchB, sizeof(float)*COLS, ROWS, cudaMemcpyHostToDevice));
		cudaEventRecord(stop);
		cudaEventSynchronize(stop); // block cpu execution till stop recorded

		// Get time taken to transfer data.
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
				deviceJacobi << <GRID, BLOCK, devPitchA >> >(B_d, A_d, ROWS, COLS, devPitchA);
				//cudaDeviceSynchronize();
			}
			// Runs first
			else {
				deviceJacobi << <GRID, BLOCK, devPitchA >> >(A_d, B_d, ROWS, COLS, devPitchB);
				//cudaDeviceSynchronize();
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
		checkCudaErrors(cudaMemcpy2D(A, hostPitchA, A_d, devPitchA, COLS*sizeof(float), ROWS, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

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

