#include<stdio.h>
#define N 1024*1024

__global__ void compare(const unsigned int* a, const unsigned int* b, double *rd, unsigned int *c) //kernel func, run on GPU
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;;    // this thread handles the data at its thread id
    //if(tid >= N) return;
    double x = rd[tid];
    // SIMD Version
    //unsigned int condition = __vsetne4(a[tid], b[tid]);
    for(std::size_t i =0; i < 1e5 ; ++i)
    {
        if(__vsetne4(a[tid], b[tid])) {
            x = 4 * x * (x-1);
        } else {
            x = 3.8 * x * (x-1);
        }
    }
    rd[tid] = x;
}
 
int cpu_compare(unsigned int *a, unsigned int *b, unsigned int *c, double *rd)
{
    unsigned int *dev_a, *dev_b, *dev_c;
    double *dev_rd;
    cudaMalloc((void**)&dev_a, N*sizeof(unsigned int)); //alloc gpu memory
    cudaMalloc((void**)&dev_b, N*sizeof(unsigned int)); 
    cudaMalloc((void**)&dev_c, N*sizeof(unsigned int));
    cudaMalloc((void**)&dev_rd, N*sizeof(double));
    cudaMemcpy(dev_a, a, N*sizeof(unsigned int), cudaMemcpyHostToDevice); // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_b, b, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_rd, rd, N*sizeof(double), cudaMemcpyHostToDevice);
    
    compare<<<N,1>>>(dev_a, dev_b, dev_rd, dev_c); //call kernel func<<<block, thread>>>
    
    cudaMemcpy(c, dev_c, N*sizeof(unsigned int), cudaMemcpyDeviceToHost); //copy back the data
    cudaMemcpy(rd, dev_rd, N*sizeof(double), cudaMemcpyDeviceToHost); //copy back the data
    double sum=0;
    for(std::size_t i=0; i < N; ++i) {
        sum+=rd[i];
    }
    printf("Sum = %lf\n",sum);
    cudaFree(dev_a); //memory free
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_rd);
    return 0;
}

int main() {
    unsigned int a[N], b[N], c[N];
    double rd[N];
    for (int i = 0; i < N; ++i) {
        a[i] = rand()%256;
        b[i] = rand()%256;
        rd[i] = ((double)rand()*1.0)/(double)RAND_MAX;
    }
    cpu_compare(a, b, c, rd);
    return 0;
}
