#include <stdio.h>


__global__ void convolution(float *input, int isize, int ichan, float *output,
        int osize, int ochan, float *weight, float *bias, int ksize, int stride, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int och, orow, ocol;
    int kch, kcol, krow;

    if (idx < N) {
        och = idx / (osize * osize);
        orow = (idx % (osize * osize)) / osize;
        ocol = (idx % (osize * osize)) % osize;
        // printf("%d, %d, %d\n", och, orow, ocol);
        *(output+och*osize*osize+orow*osize+ocol) = 0.0;
        for (krow = 0; krow < ksize; krow++) {
            for (kcol = 0; kcol < ksize; kcol++) {
                for (kch = 0; kch < ichan; kch++) {
                    // output[och][ocol][orow] += weight[och][kch][kcol][krow] * input[kch][kcol + ocol*stride][krow + orow*stride];
                    // example : conv1_out[57] += conv1_w[i*11*11+j*11+k] * image[(227*4*1+4*2)+i*227*227+j*227+k];
                    *(output+och*osize*osize+orow*osize+ocol) += *(weight+och*ichan*ksize*ksize+kch*ksize*ksize+krow*ksize+kcol) *
                    *(input+kch*isize*isize+krow*isize+kcol+(orow*isize*stride+ocol*stride));
                }
            }
        }
        *(output+och*osize*osize+orow*osize+ocol) += *(bias+och);
    }
}


__host__ void convolution(float *input, int isize, int ichan, float *output, int osize, int ochan, float *weight, float *bias, int ksize, int stride){
  /*
	Data Format:
	input[ch (< ichan)][row (< isize)][[col (< isize)]
	output[ch (< ochan)][row (< osize)][col (< osize)]
	weight[karnel (< ochan)][ch (< ichan)][row (< ksize)][col (< ksize)]
	bias[karnel (< ochan)]
   */
  int ocol, orow, och, kcol, krow, kch;

  printf("Convolution:\n");
  printf("  isize=%d, ichan=%d, osize=%d, ochan=%d, ksize=%d, stride=%d\n", isize, ichan, osize, ochan, ksize, stride);
  for (och= 0; och < ochan; och++) {
    for (orow = 0; orow < osize; orow++) {
      for (ocol = 0; ocol < osize; ocol++) {
        *(output+och*osize*osize+orow*osize+ocol) = 0.0;
        for (krow = 0; krow < ksize; krow++) {
          for (kcol = 0; kcol < ksize; kcol++) {
            for (kch = 0; kch < ichan; kch++) {
              // output[och][ocol][orow] += weight[och][kch][kcol][krow] * input[kch][kcol + ocol*stride][krow + orow*stride];
              // example : conv1_out[57] += conv1_w[i*11*11+j*11+k] * image[(227*4*1+4*2)+i*227*227+j*227+k];
              *(output+och*osize*osize+orow*osize+ocol) += *(weight+och*ichan*ksize*ksize+kch*ksize*ksize+krow*ksize+kcol) *
              *(input+kch*isize*isize+krow*isize+kcol+(orow*isize*stride+ocol*stride));
              }
            }
          }
          *(output+och*osize*osize+orow*osize+ocol) += *(bias+och);
        }
      }
    }
  printf("\n");fflush(stdout);
}
