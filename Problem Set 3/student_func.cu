/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "reference_calc.cpp"
#include "utils.h"
#include <cstdio>

__global__       
void findMin(const float* const d_logLuminance, float* d_out, const size_t size)
{
    // indexing
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if(id >= size)
    {
        return;
    }
    
    //Copy to shared memory
    extern __shared__ float s_logLum[];
    s_logLum[tid] = d_logLuminance[id];
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_logLum[tid] = min(s_logLum[tid + s],s_logLum[tid]); 
        }
        __syncthreads();
    }
    
    if (tid == 0)
    {
        d_out[blockIdx.x] = s_logLum[tid];
    }
}

__global__       
void findMax(const float* const d_logLuminance, float* d_out, const size_t size)
{
    // indexing
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if(id >= size)
    {
        return;
    }
    
    //Copy to shared memory
    extern __shared__ float s_logLum[];
    s_logLum[tid] = d_logLuminance[id];
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_logLum[tid] = max(s_logLum[tid + s],s_logLum[tid]); 
        }
        __syncthreads();
    }
    
    if (tid == 0)
    {
        d_out[blockIdx.x] = s_logLum[tid];
    }
}

__global__
void generateHistogram(const float* const d_logLuminance,
                       float min_logLum,
                       const size_t numBins, 
                       int* histogram,
                       float range)
{
    // indexing
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // determine appropriate bin
    int bin = (int)(((d_logLuminance[index] - min_logLum) / range) * (float)(numBins));

    //increment bin count
    atomicAdd(&(histogram[bin]), 1);
    
    if(bin < 16) printf("Bin = %d Count = %d\n",bin,histogram[bin]);
}

__global__
void prefixSum(unsigned int* const out, int* in, const size_t size)
{
    //indexing
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (gid >= size)
        return;
        
    //copy input array to intermediate array to start
    extern __shared__ int s_intermediate[];
    s_intermediate[tid] = in[gid];
    __syncthreads();
    
    //copy input array to intermediate array to start
    extern __shared__ int s_out[];
    s_out[tid] = s_intermediate[tid];
    __syncthreads();
    
    //0th value in the intermediate does not change
    if (tid == 0)
        out[tid] = in[tid];
    
    //Copy in[] to shared memory
    extern __shared__ int s_in[];
    s_in[tid] = in[gid];
    __syncthreads();
    
    //reduce
    for(int s = 1; s < blockDim.x; s *= 2)
    {
        int pos = (tid + 1) * s * 2 - 1;
        if (pos == blockDim.x - 1)
        {
            s_intermediate[pos] = 0;
            
            //out becomes the next in for next stride
            s_in[pos] = s_intermediate[pos];
        } 
        else if (pos < blockDim.x)
        {
            s_intermediate[pos] = s_in[pos] + s_in[pos - s];
            
            //out becomes the next in for next stride
            s_in[pos] = s_intermediate[pos];
        }
        __syncthreads();
    }
    
    //downsweep
    for (int s = blockDim.x; s > 0; s /= 2)
    {
        //get new position
        int pos = (tid + 1) * s * 2 - 1;
        
        if(pos < blockDim.x)
        {
            //swap and sum values
            int x = s_in[pos];
            int y = s_in[pos - s];
            s_out[pos] = y + x; //sum
            s_out[pos - s] = x; //swap
            
            //out becomce next in
            s_intermediate[pos] = s_out[pos];
            s_intermediate[pos - s] = s_out[pos - s];
        }
        __syncthreads();
    }
    
    //copy back to device
    out[gid] = s_out[tid]; //or s_out[tid] with downsweep
    __syncthreads();
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    
    //threads and blocks
    int threads = 1024; 
    int blocks = (numCols * numRows + threads - 1) / threads;
    printf("\nblocks = %d\nrows = %d\ncols = %d\n\n",blocks,numRows,numCols);
    
    //variables
    float *d_min_array;
    float *d_max_array;
    float *d_min_out;
    float *d_max_out;
    checkCudaErrors(cudaMalloc((void **) &d_min_array, sizeof(float) * blocks));
    checkCudaErrors(cudaMalloc((void **) &d_min_out, sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_max_array, sizeof(float) * blocks));
    checkCudaErrors(cudaMalloc((void **) &d_max_out, sizeof(float)));
    
    /////////////////////////////////find max//////////////////////////////////////////////
    findMax<<<blocks, threads, threads * sizeof(float)>>>(d_logLuminance, d_max_array, numCols * numRows);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   
    findMax<<<1, threads, threads * sizeof(float)>>>(d_max_array, d_max_out, blocks);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   
    checkCudaErrors(cudaMemcpy(&max_logLum, d_max_out, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Max = %f\n",max_logLum);
    ///////////////////////////////////////////////////////////////////////////////////////
    
    /////////////////////////find min///////////////////////////////////////////////
    findMin<<<blocks, threads, threads * sizeof(float)>>>(d_logLuminance, d_min_array, numCols * numRows);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    findMin<<<1, threads, threads * sizeof(float)>>>(d_min_array, d_min_out, blocks);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
   
    checkCudaErrors(cudaMemcpy(&min_logLum, d_min_out, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Min = %f\n",min_logLum);
    ///////////////////////////////////////////////////////////////////////////////
    
    //////////////////////calc range////////////////////////////////////////////////////
    float range = max_logLum - min_logLum;
    printf("range = %f\n\n",range);
     
    ////////////////////////histogram//////////////////////////////////////////////////////  
    //initialize histogram
    int histogram[numBins];
    for (int i = 0; i < numBins; i++)
        histogram[i] = 0;
        
    printf("Bins = %d\n\n",numBins);
        
    int* d_histo;
    checkCudaErrors(cudaMalloc((void **) &d_histo, numBins * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_histo, histogram, numBins * sizeof(int), cudaMemcpyHostToDevice));
    
    
    //make histogram
    generateHistogram<<<blocks, threads>>>(d_logLuminance,
                                           min_logLum,
                                           numBins, 
                                           d_histo,
                                           range);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    ////////////////////////////////preform exclusive sum scan on histogram///////////////////////
    blocks = (numBins + threads - 1) / threads;
    printf("\nBlocks = %d Threads = %d\n\n",blocks,threads);
    prefixSum<<<blocks, threads, threads * sizeof(int)>>>(d_cdf, d_histo, numBins);
    
    cudaFree(d_min_array);
    cudaFree(d_max_array);
    cudaFree(d_min_out);
    cudaFree(d_max_out);
    cudaFree(d_histo);
}
