//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */


__global__ 
void getPredicates(unsigned int* d_in, size_t size, int bit, unsigned int* d_false_predicate,
    unsigned int* d_true_predicate)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= size)
		return;

    unsigned int zero = !(d_in[index] & (1 << bit));
    d_false_predicate[index] = zero;
    d_true_predicate[index] = !zero;
	
	if(index < 16)
		printf(" %d", d_false_predicate[index]);
}

__global__
void sumReduce(unsigned int* sum, unsigned int* input, size_t size)
{
    //indexing
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (gid >= size)
        return;
    
    //copy input array to shared
    extern __shared__ unsigned int s_sums[];
    s_sums[tid] = input[gid];
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_sums[tid] += s_sums[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0)
    {
        sum[blockIdx.x] = s_sums[tid];
    }
}

__global__
void prefixSum(unsigned int* out, unsigned int* in, size_t size)
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
        if (pos == blockDim.x - 1) //last element is 0
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
	
	//print
    if (gid < 16)
        printf("%d ", out[gid]);
}

__global__
void prefixSumFix(unsigned int* scan, unsigned int* sums, size_t numElems)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= numElems)
        return;
    
    scan[index] += sums[blockIdx.x];
    
    //print
    if (index > 1023 && index < 1040)
        printf("%d ", scan[index]);
}

__global__
void scatterElements(unsigned int* d_inputVals, unsigned int* d_outputVals, unsigned int* true_scan, 
                     unsigned int* false_scan, unsigned int* d_inputPos, unsigned int* d_outputPos, 
                     size_t numElems, unsigned int bit)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= numElems)
        return;
    
    unsigned int isLastElemZero = ((d_inputVals[numElems - 1] & (1 << bit)) == 0);
    unsigned int oneStartPos = false_scan[numElems - 1] + isLastElemZero;
    
    unsigned int pos = d_inputPos[index];
    unsigned int value = d_inputVals[index];
    int new_loc;
    
    if (value & (1 << bit))
    {
        new_loc = oneStartPos + true_scan[index];    
        d_outputVals[new_loc] = value;
        d_outputPos[new_loc] = pos;
    }
    else
    {
        new_loc = false_scan[index];
        d_outputVals[new_loc] = value;
        d_outputPos[new_loc] = pos;
    }
    
    //print example
    if (index == 0)
        printf("isLastElemZero = %d oneStartPos = %d NewLoc = %d Value = %d\n", isLastElemZero, oneStartPos, 
                new_loc, value);
}

__global__ 
void copyBuffers(unsigned int* d_inVals, unsigned int* d_inPos, unsigned int* d_outVals, unsigned int* d_outPos,
                 size_t numElems)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numElems)
        return;
    
    d_outVals[index] = d_inVals[index];
    d_outPos[index] = d_inPos[index];
}

void your_sort(unsigned int* d_inputVals,
               unsigned int* d_inputPos,
               unsigned int* d_outputVals,
               unsigned int* d_outputPos,
               const size_t numElems)
{
    //kernal threads and blocks
    int threads = 1024;
    int blocks = (numElems + threads - 1) / threads;
    printf("numElems = %d\nBlocks = %d\n\n", numElems, blocks);
    
    //initialize buffers
    unsigned int* d_true_predicate; 
    checkCudaErrors(cudaMalloc((void **) &d_true_predicate, numElems * sizeof(unsigned int)));
    unsigned int* d_true_sums;
    checkCudaErrors(cudaMalloc((void **) &d_true_sums, blocks * sizeof(unsigned int)));
    unsigned int* d_true_sums_scan;
    checkCudaErrors(cudaMalloc((void **) &d_true_sums_scan, blocks * sizeof(unsigned int)));
    unsigned int* d_true_scan;
    checkCudaErrors(cudaMalloc((void **) &d_true_scan, numElems * sizeof(unsigned int)));
    
    unsigned int* d_false_predicate;
    checkCudaErrors(cudaMalloc((void **) &d_false_predicate, numElems * sizeof(unsigned int)));
    unsigned int* d_false_sums;
    checkCudaErrors(cudaMalloc((void **) &d_false_sums, blocks * sizeof(unsigned int)));
    unsigned int* d_false_sums_scan;
    checkCudaErrors(cudaMalloc((void **) &d_false_sums_scan, blocks * sizeof(unsigned int)));
    unsigned int* d_false_scan;
    checkCudaErrors(cudaMalloc((void **) &d_false_scan, numElems * sizeof(unsigned int)));
    
    //loop thru bits 
    for(unsigned int bit = 0; bit < 8 * sizeof(unsigned int); bit++)
    {
        printf("\nBit position = %d\n", bit);
        
        //0 all buffers everytime
        cudaMemset(d_true_predicate, 0, numElems * sizeof(unsigned int));
        cudaMemset(d_true_scan, 0, numElems * sizeof(unsigned int));
        cudaMemset(d_true_sums, 0, blocks * sizeof(unsigned int));
        cudaMemset(d_true_sums_scan, 0, blocks * sizeof(unsigned int));
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		
        cudaMemset(d_false_predicate, 0, numElems * sizeof(unsigned int));
        cudaMemset(d_false_scan, 0, numElems * sizeof(unsigned int));
        cudaMemset(d_false_sums, 0, blocks * sizeof(unsigned int));
        cudaMemset(d_false_sums_scan, 0, blocks * sizeof(unsigned int));
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        
		//get predicates
		printf("False predicate(first 16): ");
		getPredicates<<<blocks, threads>>>(d_inputVals, numElems, bit, d_false_predicate, d_true_predicate);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		printf("\n");
				
		//get block sums
		printf("False sums(first 16): ");
		sumReduce<<<blocks, threads, threads * sizeof(unsigned int)>>>(d_false_sums, d_false_predicate, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		printf("\nTrue sums(first 16): ");
		sumReduce<<<blocks, threads, threads * sizeof(unsigned int)>>>(d_true_sums, d_true_predicate, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		printf("\n");
		
		//scan blockwise
		printf("False scan(first 16): ");
		prefixSum<<<blocks, threads, threads * sizeof(unsigned int)>>>(d_false_scan, d_false_predicate, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		printf("\nTrue scan(first 16): ");
		prefixSum<<<blocks, threads, threads * sizeof(unsigned int)>>>(d_true_scan, d_true_predicate, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		printf("\n");
		
		//scan block sums
		printf("False sums scan(first 16): ");
		prefixSum<<<1, threads, threads * sizeof(unsigned int)>>>(d_false_sums_scan, d_false_sums, blocks);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		printf("\nTrue sums scan(first 16): ");
		prefixSum<<<1, threads, threads * sizeof(unsigned int)>>>(d_true_sums_scan, d_true_sums, blocks);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		printf("\n");
		
		//fix scans
		printf("False scan fixed(After element 1023): ");
		prefixSumFix<<<blocks, threads>>>(d_false_scan, d_false_sums_scan, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		printf("\nTrue scan fixed(After element 1023): ");
		prefixSumFix<<<blocks, threads>>>(d_true_scan, d_true_sums_scan, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		printf("\n");
		
		//scatter
        printf("Scatter Element Example:\n");
        scatterElements<<<blocks, threads>>>(d_inputVals, d_outputVals, d_true_scan, d_false_scan, d_inputPos, 
                                             d_outputPos, numElems, bit);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		
        //swap buffers (pointers)
        std::swap(d_outputVals, d_inputVals);
        std::swap(d_outputPos, d_inputPos);
        
        printf("\n_________________________________________________________________________________________\n");
        //return;
    }
    
    //copy from input buffer to output
    copyBuffers<<<blocks, threads>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
    //delete buffers
    cudaFree(d_true_predicate);
    cudaFree(d_true_sums);
    cudaFree(d_true_sums_scan);
    cudaFree(d_true_scan);
    cudaFree(d_false_predicate);
    cudaFree(d_false_sums);
    cudaFree(d_false_sums_scan);
    cudaFree(d_false_scan);
}
