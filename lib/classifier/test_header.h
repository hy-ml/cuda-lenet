__host__ void classifier(float *input, int isize, float *output, int osize,
                         float *weight, float *bias);
__global__ void classifier(float *input, int isize, float *output, int osize,
                           float *weight, float *bias, int N);
