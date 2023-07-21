#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/random.h>
#include <time.h>
#include <cuZFP.h>

struct pcg_state_impl {
	uint64_t state;
	uint64_t stream;
};

typedef struct pcg_state_impl pcg_state;

uint32_t pcg32_random(pcg_state* rng)
{
	const uint64_t old = rng->state;
	//Advance internal state
	rng->state = (rng->state) * 0X5851F42D4C957F2DULL;
	// NOLINTNEXTLINE(hicpp-signed-bitwise)
	rng->state += (rng->stream | 1);
	const uint32_t xorshifted = ((old >> 18U) ^ old) >> 27U;
	const uint32_t rot = old >> 59U;
	// NOLINTNEXTLINE(hicpp-signed-bitwise)
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

int soma_seed_rng(pcg_state* rng, uint64_t seed, uint64_t stream)
{
	rng->stream = stream * 2 + 1;
	rng->state = 0;
	pcg32_random(rng);
	rng->state += seed;
	pcg32_random(rng);
	// Improve quality of first random numbers
	pcg32_random(rng);
	return 0;
}

int main() 
{	
	// initial data in host and device
	size_t size = 1000000;
	size_t byte_size = size * sizeof(double);

	double* a_host = (double*)malloc(byte_size);
	double* b_host = (double*)malloc(byte_size);
	double* c_host = (double*)malloc(byte_size);

	if(a_host == NULL || c_host == NULL) {
		fprintf(stderr, "Failed to allocate host vectors.\n");
		return EXIT_FAILURE;
	}
	uint64_t seed = 0;
	uint64_t stream = 0;
	pcg_state state;

	getrandom(&seed, sizeof(uint64_t), 0);
	soma_seed_rng(&state, seed, stream);

	for(size_t i=0; i<size; ++i) {
		a_host[i] = pcg32_random(&state) / (double)UINT32_MAX;
	}

	double* a_device = NULL;
	double* b_device = NULL;
	double* c_device = NULL;

	cudaError_t error_code_a = cudaMalloc((void**)&a_device, byte_size);
	//cudaError_t error_code_b = cudaMalloc((void**)&b_device, byte_size);
	cudaError_t error_code_c = cudaMalloc((void**)&c_device, byte_size);

	if (error_code_a != cudaSuccess || error_code_c != cudaSuccess) {
		fprintf(stderr, "Failed to allocate to device vector.\n");
		return EXIT_FAILURE;
	}

	cudaError_t error_code_amemc = cudaMemcpy(a_device, a_host, byte_size, cudaMemcpyHostToDevice);
	if(error_code_amemc != cudaSuccess) {
		fprintf(stderr, "Failed to copy memory from host to device");
	}
	
	//compress, decompress
	zfp_stream* zfp;
	bitstream* bstream;
	bitstream* cstream;
	zfp_field *in_field = zfp_field_1d(a_device, zfp_type_double, size);

	int rate = 32;
	int dims = 1;
	zfp = zfp_stream_open(NULL);
	zfp_stream_set_rate(zfp, rate, in_field->type, dims, zfp_false);
	size_t sfpsize;
	size_t buffsize = zfp_stream_maximum_size(zfp, in_field);
	fprintf(stderr, "byte_size = %d\t", byte_size);
	fprintf(stderr, "buffsize = %d\t", buffsize);
	cudaError_t error_code_b = cudaMalloc((void**)&b_device, buffsize);
	bstream = stream_open(b_device, buffsize);
	zfp_stream_set_bit_stream(zfp, bstream);
	//cuda_compress(zfp, in_field);
	if (zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
		zfpsize = zfp_compress(zfp, in_field);
	} else {
		fprintf(stderr, "Failed to compress");
	}
	stream_close(bstream);
	zfp_field_free(in_field);

	cstream = stream_open(b_device, buffsize);
	zfp_stream_set_bit_stream(zfp, cstream);
	zfp_field *out_field;
	out_field = zfp_field_1d(c_device, zfp_type_double, size);
	cuda_decompress(zfp, out_field);
	if (zfp_stream_set_execution(zfp, zfp_exec_cuda)) {                 
		zfpsize = zfp_decompress(zfp, out_field);
       		fprintf(stderr, "sfpsize = %d\n", zfpsize);		
	} else {
		 fprintf(stderr, "Failed to decompress");
	}
	stream_close(cstream);
	zfp_field_free(out_field);
	zfp_stream_close(zfp);

	cudaError_t error_code_cmemc = cudaMemcpy(c_host, c_device, byte_size, cudaMemcpyDeviceToHost);
	cudaError_t error_code_bmemc = cudaMemcpy(b_host, b_device, buffsize, cudaMemcpyDeviceToHost);
	for(size_t i = 0; i < size; ++i) {
		if(a_host[i] - c_host[i] > pow(10,-6))
			fprintf(stdout, "a_host[%d] = %f, c_host[%d] = %f ", i, a_host[i], i, c_host[i]);
	}
	return EXIT_SUCCESS;
}
