#include <stdio.h>


__host__ void maxpooling(float *input, int isize, int ichan, float *output, int osize,  int ksize, int stride) {
  int ocol, orow, och, kcol, krow;
  float max, tmp;

  printf("MaxPooling:\n");
  printf("  isize=%d, ichan=%d, osize=%d, ksize=%d, stride=%d\n", isize, ichan, osize, ksize, stride);

  for (och= 0; och < ichan; och++) {
	for (orow = 0; orow < osize; orow++) {
	  for (ocol = 0; ocol < osize; ocol++) {
		max = -256.0;
		for (krow = 0; krow < ksize; krow++) {
		  for (kcol = 0; kcol < ksize; kcol++) {
			tmp = *(input+och*isize*isize+krow*isize+kcol+(orow*isize*stride+ocol*stride));
			//tmp = input[och][orow+krow][ocol+kcol];
			if (max < tmp) max = tmp;
		  }
		*(output+och*osize*osize+osize*orow+ocol) = max;
		}
	  }
	}
  }
  printf("\n");fflush(stdout);
}


__global__ void maxpooling(float *input, int isize, int ichan, float *output,
      int osize,  int ksize, int stride, int N) {
    int ocol, orow, och, kcol, krow;
    float max, tmp;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        och = idx / (osize * osize);
        orow = (idx % (osize * osize)) / osize;
        ocol = (idx % (osize * osize)) % osize;
        max = -256.0;
        for (krow = 0; krow < ksize; krow++) {
            for (kcol = 0; kcol < ksize; kcol++) {
                tmp = *(input+och*isize*isize+krow*isize+kcol+(orow*isize*stride+ocol*stride));
                //tmp = input[och][orow+krow][ocol+kcol];
                if (max < tmp) max = tmp;
            }
            *(output+och*osize*osize+osize*orow+ocol) = max;
        }
    }
  }