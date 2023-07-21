# result_record

repo for recording the running/profiling result

For the structure of this folder, refer to the file struct.txt

##### compiling WalBerla

NOTE: I'm sorry to say, I failed to compiling this software in unitn cluster due to we are modifying the version which is still in develop, some of the environment is not easy to build. I can offer all the profiling results(ncu and nsys) for the origin software while all the mini-app and toy case is offered in this repo.  

pre-requirement

```shell
CMAKE_VERSION="3.21.1"
CUDA_VERSION="11.5"
GCC_VERSION="11.2.0"
PSMPI_VERSION="5.5.0-1"
PYTHON_VERSION="3.9.6"
LBMPY_VERSION="master"
PYSTENCILS_VERSION="master"
OPENMESH_VERSION="8.1"
```

(may not work now since we are porting the code into new env.....)

scriptal : [Scalable / Scriptal · GitLab (fz-juelich.de)](https://gitlab.jsc.fz-juelich.de/scalable/scriptal)

download the showcases rep: https://gitlab.jsc.fz-juelich.de/scalable/showcases.git

NOTE: THIS SCRIPT ONLY WORKS ON JUWELS AND JURECA NOW DUE TO THE ENV SRC FILE : scriptal.walberla.env

The script for general-gpu is not yet valid

compile walberla using scalable:

```shell
WALBERLA_ENV_VERSION=juwels-booster.cuda
mkdir env-$WALBERLA_ENV_VERSION
mkdir install-$WALBERLA_ENV_VERSION
git clone https://i10git.cs.fau.de/walberla/walberla.git
./src/bin/scriptal.walberla.env -e $WALBERLA_ENV_VERSION -p  $(pwd)/env-$WALBERLA_ENV_VERSION/ -t $(pwd)/tmp
mkdir build && cd build
cmake -DWALBERLA_BUILD_WITH_PYTHON=On    -DWALBERLA_BUILD_WITH_GPU_SUPPORT=On -DWALBERLA_BUILD_WITH_CODEGEN=On   -DWALBERLA_BUILD_WITH_CUDA=On ../walberla/
make -j24 UniformGridGPU_d3q27_pull_srt
srun --nodes=1 --time=00:10:00 --gres=gpu:4 --account=scalable2022 --partition=booster ./apps/benchmarks/UniformGridGPU/UniformGridGPU_d3q27_pull_srt ./apps/benchmarks/UniformGridGPU/simulation_setup/benchmark_configs.py
```

compile showcase using scalable

```shell
source /p/project/scalable2022/zheng3/scriptal/env-juwels-booster.cuda/bin/activate
cmake -DWALBERLA_DIR=../walberla .. -DWALBERLA_BUILD_WITH_CUDA=On  -DWALBERLA_BUILD_WITH_OPENMESH=On  -DWALBERLA_BUILD_WITH_CODEGEN=On
make S2A
srun --account=scalable2022 --nodes=1 --time=00:10:00 --gres=gpu:4 ./S2A S2A_CPU_weak_scaling.prm  --partition=develbooster
srun --account=scalable2022 --nodes=1 --time=00:10:00 --gres=gpu:4 nsys profile ./S2A S2A_CPU_weak_scaling.prm  --partition=develbooster
```

##### when need to run self-modificated code:

INABLE(delete) THE S2A AUTO-GENERATION in cmakelist

cover the origin code with the new one, then re-compile using the command `make S2A`

##### compile the zfplib with CUDA

```shell
git clone https ://github.com/LLNL/zfp.git
cd zfp
mkdir build
cd build
cmake −DZFP WITH CUDA=ON ..
cmake −−build . −−config Release
```

export the LIB

```shell
export LD LIBRARY PATH=
$LD LIBRARY PATH:/p/home/jusers/zheng3/juwels/comptest/zfp/build/lib64

// compiling example
nvcc cuda random.cu −I ../include −L ../ build /lib64 −lzfp −lm
```

Modify the CMakeLists.txt for UniformGridGPU

```shell
find package (zfp)
target link libraries (UniformGridxyz PRIVATE zfp::zfp)
```

set the head file for zfp

```shell
target include directories (cuda PRIVATE /p/home/jusers/zheng3/juwels/comptest/zfp/include)
```

set the zfp dir when compiling with cmake

```shell
cmake −DCMAKE PREFIX PATH=/p/home/jusers/zheng3/juwels/comptest/zfp/build −DWALBERLA BUILD WITH PYTHON=On
−DWALBERLA BUILD WITH GPU SUPPORT=On −DWALBERLA BUILD WITH CODEGEN=On
−DWALBERLA BUILD WITH CUDA=On ../walberla
```

##### A sample script for running the code in JURECA-DC/JUWELS-BOOSTER

```
#!/bin/bash
#SBATCH -A paj2305
#SBATCH -N 4
#SBATCH -t 240
#SBATCH -p dc-gpu-devel
#SBATCH -G 16

srun --account=paj2305 --nodes=4 --time=01:00:00 --gres=gpu:4 ncu  --target-processes all --export perf_nosimd.report --import-source=yes    --metrics regex:sm__inst_executed_pipe_*,regex:sm__sass_thread_inst_executed_op*  --page raw --set full ./S2A S2A_CPU_weak_scaling.prm  --partition=dc-gpu-devel

```

