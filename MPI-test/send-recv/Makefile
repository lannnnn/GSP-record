EXECS=sendRecv
MPICC?=mpicc

all: ${EXECS}

sendRecv: sendRecv.c
		${MPICC} -o sendRecv sendRecv.c

ping_pong: ping_pong.c
		${MPICC} -o ping_pong ping_pong.c

ring: ring.c
		${MPICC} -o ring ring.c

clean:
		rm -f ${EXECS}
