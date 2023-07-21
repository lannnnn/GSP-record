#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int world_size;     
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Request handle;
    clock_t start, finish;

    int N = 10;
    if (argc >= 2) {
    //    fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
        N = atoi(argv[1]);
    }

    double number[N];
    for(int i=0; i<N; i++) {
	srand((unsigned) time(0));
	number[i] = (1.0 / RAND_MAX) * rand(); 
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = clock();

    if(world_rank == 0) {
	MPI_Request sendHandle;
	MPI_Isend(&number, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &sendHandle);
	MPI_Wait(&sendHandle, MPI_STATUS_IGNORE);
	finish = clock();
	printf("Send node waiting time = %f seconds\n", (double)(finish-start)/CLOCKS_PER_SEC);
    } else if(world_rank == 1){
    	float recv_data[N];
	MPI_Request recvHandle;
	MPI_Irecv(&recv_data, N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &recvHandle);
	MPI_Wait(&recvHandle, MPI_STATUS_IGNORE);
	printf("received %d data[N-1] = %f\n ", N, (double)recv_data[N-1]);
    }
	
    //MPI_Barrier(MPI_COMM_WORLD);
    //finish = clock();

    //if(world_rank == 0)
    //    printf("Elapsed time = %f seconds\n", (double)(finish-start)/CLOCKS_PER_SEC);

    MPI_Finalize();
}
