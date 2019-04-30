/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "reference.cpp"
#include <stdio.h>

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
    //indexing
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= numVals)
        return;
    
    //increment bin
    int bin = vals[index];
    atomicAdd(&(histo[bin]), 1);
    
    /*if (bin == 119)
        printf("Val = %d Bin = %d count = %d\n", vals[index], bin, histo[bin]);*/
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
    //Variables
    printf("numElems = %d ", numElems);
    printf("numBins = %d ", numBins);
        
    //threads and blocks
    int blockWidth = 1024;
    int blocks = (numElems + blockWidth - 1) / blockWidth;
    printf("threads per block = %d ", blockWidth);
    printf("blocks = %d\n", blocks);
    
    dim3 blockSize(blockWidth, 1, 1); //threads per block
    dim3 gridSize(blocks, 1, 1); // number of blocks
    
    //generate histogram
    yourHisto<<<gridSize, blockSize>>>(d_vals, d_histo, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    /*//copy to host and print
    unsigned int h_histo[numBins];
    for (int i = 0; i < numBins; i++)
        h_histo[i] = 0;
    cudaMemcpy(h_histo, d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++)
        printf(" %d", h_histo[i]);
    printf("\n");*/
}
