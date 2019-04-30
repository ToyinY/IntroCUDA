//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"
#include <cstdio>

__global__
void createMask(uchar4* d_sourceImg, uchar4* mask, size_t numRows, size_t numCols)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int index = y * numCols + x;
    if ((x >= numCols) || (y >= numRows))
        return;
        
    //get pixel color
    unsigned char red = d_sourceImg[index].x;
    unsigned char green = d_sourceImg[index].y;
    unsigned char blue = d_sourceImg[index].z;
    int value = red + green + blue;
    int white_value = 255 * 3;
        
    //add to mask if color is not white
    if (value < white_value)
    {
        mask[index].x = red;
        mask[index].y = green;
        mask[index].z = blue;
    } else {
        mask[index].x = 255;
        mask[index].y = 255;
        mask[index].z = 255;
    }
}

__global__
void getInteriorAndBorderRegions(int* d_interiorPixels, int* d_borderPixels, uchar4 * d_mask, size_t numRows, 
                                 size_t numCols)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int index = y * numCols + x;
    if ((x >= numCols) || (y >= numRows))
        return;
    
    //4 surrounding pixels
    int up = (y + 1) * numCols + x;
    int down = (y - 1) * numCols + x;
    int right = y * numCols + x + 1;
    int left = y * numCols + x - 1;
        
    //get current pixel value
    unsigned char red = d_mask[index].x;
    unsigned char green = d_mask[index].y;
    unsigned char blue = d_mask[index].z;
    int value = red + green + blue;
    int white_value = 255 * 3;
    
    //must not be white (be in the mask)
    if (value != white_value)
    {
        //interior
        if ((d_mask[up].x + d_mask[up].y + d_mask[up].z != white_value) && 
            (d_mask[down].x + d_mask[down].y + d_mask[down].z != white_value) && 
            (d_mask[right].x + d_mask[right].y + d_mask[right].z != white_value) && 
            (d_mask[left].x + d_mask[left].y + d_mask[left].z != white_value))
        {
            d_interiorPixels[index]++;
        } 
        
        //the border
        else 
        { 
            d_borderPixels[index]++;
        }
    }
}

__global__
void separateChannels(uchar4* image, float* redChannel, float* greenChannel, float* blueChannel, size_t numRows, 
                      size_t numCols)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int index = y * numCols + x;
	if ((x >= numCols) || (y >= numRows))
	    return;
	
    //get rgb values from image and assign to new channels
	uchar4 color = image[index];
	redChannel[index] = color.x;
	greenChannel[index] = color.y;
	blueChannel[index] = color.z;
}

__global__
void recombineChannels(float* redChannel, float* greenChannel, float* blueChannel,
                       uchar4* outputImageRGBA, size_t numRows, size_t numCols)
{
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int index = y * numCols + x;
	if ((x >= numCols) || (y >= numRows))
	    return;
	
    unsigned char red   = redChannel[index];
    unsigned char green = greenChannel[index];
    unsigned char blue  = blueChannel[index];
    
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);
    
    outputImageRGBA[index] = outputPixel;
}

__global__
void jacobi(float* guess_1, float* guess_2, int* d_borderPixels, int* d_interiorPixels, float* d_inputChannel, 
            float* d_outputChannel, size_t numRows, size_t numCols)
{
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int index = y * numCols + x;
	if ((x >= numCols) || (y >= numRows))
	    return;

	unsigned int neighbor;
	
	if(d_interiorPixels[index] == 1) {
		float a = 0.f, b = 0.f, c = 0.f, d = 0.f;
		float sourceVal = d_inputChannel[index];

        //above current pixel
		if((x < numCols) && ((y + 1) < numRows)) 
		{
			d++;
			neighbor = (y + 1) * numCols + x;
			
			if(d_interiorPixels[neighbor] == 1) 
			{
				a += guess_1[neighbor];
			} 
			
			else if (d_borderPixels[neighbor] == 1) 
			{
				b += (float)d_outputChannel[neighbor];
			}
			
			c += (sourceVal - (float)d_inputChannel[neighbor]);
		}
		
		//below current pixel
		if((x < numCols) && ((y - 1) < numRows && (y - 1) > 0)) {
			d++;
			neighbor = (y - 1) * numCols + x;
			
			if(d_interiorPixels[neighbor] == 1) 
			{
				a += guess_1[neighbor];
			} 
			
			else if (d_borderPixels[neighbor] == 1) 
			{
				b += (float)d_outputChannel[neighbor];
			}
			
			c += (sourceVal - (float)d_inputChannel[neighbor]);
		}
		
		//right of current pixel
		if(((x + 1) < numCols) && (y < numRows)) 
		{
			d++;
			neighbor = y * numCols + x + 1;
			
			if(d_interiorPixels[neighbor] == 1)
			{
				a += guess_1[neighbor];
			} 
			
			else if (d_borderPixels[neighbor] == 1) 
			{
				b += (float)d_outputChannel[neighbor];
			}
			
			c += (sourceVal - (float)d_inputChannel[neighbor]);
		}
		
		//left of current pixel
		if(((x - 1) < numCols && (x - 1) > 0) && (y < numRows)) 
		{
			d++;
			neighbor = y * numCols + x - 1;
			
			if(d_interiorPixels[neighbor] == 1) 
			{
				a += guess_1[neighbor];
			} 
			
			else if (d_borderPixels[neighbor] == 1) 
			{
				b += (float)d_outputChannel[neighbor];
			}
			
			c += (sourceVal - (float)d_inputChannel[neighbor]);
		}
		
		guess_2[index] = min(255.f, max(0.0, (a + b + c) / d));
	} 
	
	else 
	{
		guess_2[index] = (float)d_outputChannel[index];
	}
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
    //get number of pixels in the image
    size_t size = numRowsSource * numColsSource;
    printf("Rows = %d cols = %d size = %d\n", numRowsSource, numColsSource, size);

    //threads and blocks
    int blockWidth = 32;
    dim3 blockSize(blockWidth, blockWidth);
    int blocksX = (numColsSource + blockSize.x - 1) / blockWidth;
    int blocksY = (numRowsSource + blockSize.y - 1) / blockWidth;
    dim3 gridSize(blocksX, blocksY);
    printf("blocksX = %d blocksY = %d\n", blocksX, blocksY);

    //initialize device images
    uchar4* d_sourceImg;
    uchar4* d_destImg;
    uchar4* d_blendedImg;
    
    //allocate memory
    checkCudaErrors(cudaMalloc(&d_sourceImg, size * sizeof(uchar4)));
    checkCudaErrors(cudaMalloc(&d_destImg, size * sizeof(uchar4)));
    checkCudaErrors(cudaMalloc(&d_blendedImg, size * sizeof(uchar4)));
    
    //copy to device
    checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, size * sizeof(uchar4), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_destImg, h_destImg, size * sizeof(uchar4), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMemset(d_blendedImg, 0, size * sizeof(uchar4)));
    
    /**
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
	**/
    
    //initialize mask    
    uchar4* d_mask;
    checkCudaErrors(cudaMalloc(&d_mask, size * sizeof(uchar4)));
    
    //create mask
    createMask<<<gridSize, blockSize>>>(d_sourceImg, d_mask, numRowsSource, numColsSource);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    /**
     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.
	**/
    
    //get interior and border regions
    int* d_interiorPixels;
    int* d_borderPixels;
    checkCudaErrors(cudaMalloc(&d_interiorPixels, size * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_borderPixels, size * sizeof(int)));
    
    //0 to start
    checkCudaErrors(cudaMemset(d_interiorPixels, 0, size * sizeof(int)));
    checkCudaErrors(cudaMemset(d_borderPixels, 0, size * sizeof(int)));
    
    getInteriorAndBorderRegions<<<gridSize, blockSize>>>(d_interiorPixels, d_borderPixels, d_mask, numRowsSource, 
                                                         numColsSource);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    /**
     3) Separate out the incoming image into three separate channels
	**/
	
    float* d_inputRedChannel;
    float* d_inputGreenChannel;
    float* d_inputBlueChannel;
    
    float* d_outputRedChannel;
    float* d_outputGreenChannel;
    float* d_outputBlueChannel;
    
    checkCudaErrors(cudaMalloc(&d_inputRedChannel, size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_inputGreenChannel, size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_inputBlueChannel, size * sizeof(float)));
    
    checkCudaErrors(cudaMalloc(&d_outputRedChannel, size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_outputGreenChannel, size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_outputBlueChannel, size * sizeof(float)));
    
    separateChannels<<<gridSize, blockSize>>>(d_sourceImg, d_inputRedChannel, d_inputGreenChannel, 
                                              d_inputBlueChannel, numRowsSource, numColsSource);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    separateChannels<<<gridSize, blockSize>>>(d_destImg, d_outputRedChannel, d_outputGreenChannel, 
                                              d_outputBlueChannel, numRowsSource, numColsSource);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    /** 
     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.
	**/

    //initialize guess buffers (will become the output)   
    float *red_1, *red_2; 
    float *green_1, *green_2; 
    float *blue_1, *blue_2;
    
    //allocate
    checkCudaErrors(cudaMalloc(&red_1, size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&red_2, size * sizeof(float)));    
    checkCudaErrors(cudaMalloc(&green_1, size * sizeof(float)));    
    checkCudaErrors(cudaMalloc(&green_2, size * sizeof(float)));    
    checkCudaErrors(cudaMalloc(&blue_1, size * sizeof(float)));    
    checkCudaErrors(cudaMalloc(&blue_2, size * sizeof(float)));
    
    //set as input image
    checkCudaErrors(cudaMemcpy(red_1, d_inputRedChannel, size * sizeof(float), cudaMemcpyDeviceToDevice));
  	checkCudaErrors(cudaMemcpy(green_1, d_inputGreenChannel, size * sizeof(float), cudaMemcpyDeviceToDevice));
  	checkCudaErrors(cudaMemcpy(blue_1, d_inputBlueChannel, size * sizeof(float), cudaMemcpyDeviceToDevice));

    /**
     5) For each color channel perform the Jacobi iteration described 
        above 800 times.
	**/
	
	for(int i = 0; i < 800; i++) {
		jacobi<<<gridSize, blockSize>>>(red_1, red_2, d_borderPixels, d_interiorPixels, d_inputRedChannel,
			                            d_outputRedChannel, numRowsSource, numColsSource);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		std::swap(red_1, red_2);
 
		jacobi<<<gridSize, blockSize>>>(green_1, green_2, d_borderPixels, d_interiorPixels, d_inputGreenChannel,
			                            d_outputGreenChannel, numRowsSource, numColsSource);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		std::swap(green_1, green_2);

		jacobi<<<gridSize, blockSize>>>(blue_1, blue_2, d_borderPixels, d_interiorPixels, d_inputBlueChannel,
			                            d_outputBlueChannel, numRowsSource, numColsSource);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		std::swap(blue_1, blue_2);
	}

    /**
     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.
	**/

    //combine channels to blended img
    recombineChannels<<<gridSize, blockSize>>>(red_1, green_1, blue_1, d_blendedImg, numRowsSource, numColsSource);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    //copy to the host
    checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, size * sizeof(uchar4), cudaMemcpyDeviceToHost));

    /*uchar4* h_reference = new uchar4[size];
    reference_calc(h_sourceImg, numRowsSource, numColsSource,
                   h_destImg, h_reference);

    checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * size, 2, .01);
    delete[] h_reference;*/

    //delete buffers
    checkCudaErrors(cudaFree(d_sourceImg));
    checkCudaErrors(cudaFree(d_destImg));
    checkCudaErrors(cudaFree(d_blendedImg));
    checkCudaErrors(cudaFree(d_mask));
    
    checkCudaErrors(cudaFree(d_interiorPixels));
    checkCudaErrors(cudaFree(d_borderPixels));
    
    checkCudaErrors(cudaFree(d_inputRedChannel));
    checkCudaErrors(cudaFree(d_inputGreenChannel));
    checkCudaErrors(cudaFree(d_inputBlueChannel));
    
    checkCudaErrors(cudaFree(d_outputRedChannel));
    checkCudaErrors(cudaFree(d_outputGreenChannel));
    checkCudaErrors(cudaFree(d_outputBlueChannel));
    
    checkCudaErrors(cudaFree(red_1));
    checkCudaErrors(cudaFree(red_2));
    checkCudaErrors(cudaFree(green_1));
    checkCudaErrors(cudaFree(green_2));
    checkCudaErrors(cudaFree(blue_1));
    checkCudaErrors(cudaFree(blue_2));
}
