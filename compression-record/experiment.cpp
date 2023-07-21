#include <cmath>
#include <fstream>
#include <mpi.h>
#include <stdio.h>
#include <vector>

using send_type = double;
auto mpi_type = MPI_DOUBLE;

double index_to_x(std::size_t i, double xl, double dx)
{
  return xl + i * dx;
}

double flux(double u)
{
  return 2 * u;
}

double numflux(double ul, double ur)
{
  return flux(ul) + flux(ur) - 2 * (ur - ul);
}

int main(int argc, char** argv)
{
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::size_t ncell = 1'000;
  // TODO: Change nhalo to /2, /4, /8, /16
  std::size_t nhalo = ncell / 16;
  double xl = -2.0;
  double xr = 2.0;
  double dx = (xr - xl) / (2 * ncell);
  double lxl = -2.0;
  double lxr = 0.0;
  double lxc = -1.0;
  double rxl = 0.0;
  double rxr = 2.0;
  double rxc = 1.0;

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::vector<double> data(ncell);
  std::vector<double> tmp(ncell);
  std::vector<send_type> halo_send(nhalo);
  std::vector<send_type> halo_recv(nhalo);

  if (world_rank == 0) {
    for (std::size_t i = 0; i < ncell; ++i) {
      auto x = index_to_x(i, lxl, dx);
      data[i] = 1 + std::exp(-10 * (x - lxc) * (x - lxc));
      tmp[i] = 1 + std::exp(-10 * (x - rxc) * (x - rxc));
    }
  } else {
    for (std::size_t i = 0; i < ncell; ++i) {
      auto x = index_to_x(i, rxl, dx);
      data[i] = 1 + std::exp(-10 * (x - rxc) * (x - rxc));
      tmp[i] = 1 + std::exp(-10 * (x - rxc) * (x - rxc));
    }
  }

  auto const filename = "data_" + std::to_string(world_rank) + ".0.txt";
  std::ofstream file(filename);
  for (std::size_t i = 0; i < ncell; ++i) {
    file << index_to_x(i, xl, dx) << " " << data[i] << "\n";
  }

  auto dt = dx / 8;

  double time = 0;
  double Tfinal = 0.20;
  std::size_t iter = 0;
  std::size_t max_iter = 100000;

  // TODO: clock start here
  while (time < Tfinal and iter < max_iter) {
    dt = std::min(dt, Tfinal - time);

    if (world_rank == 0) {
#pragma omp parallel for
      for (std::size_t i = 0; i < nhalo; ++i) {
        halo_send[i] = static_cast<send_type>(data[ncell - nhalo + i]);
      }
      MPI_Request send_handle;
      MPI_Request recv_handle;
      MPI_Irecv(halo_recv.data(), nhalo, mpi_type, 1, 0, MPI_COMM_WORLD, &recv_handle);
      MPI_Isend(halo_send.data(), nhalo, mpi_type, 1, 1, MPI_COMM_WORLD, &send_handle);
      MPI_Wait(&send_handle, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_handle, MPI_STATUS_IGNORE);

    } else {
#pragma omp parallel for
      for (std::size_t i = 0; i < nhalo; ++i) {
        halo_send[i] = static_cast<send_type>(data[i]);
      }
      MPI_Request send_handle;
      MPI_Request recv_handle;
      MPI_Irecv(halo_recv.data(), nhalo, mpi_type, 0, 1, MPI_COMM_WORLD, &recv_handle);
      MPI_Isend(halo_send.data(), nhalo, mpi_type, 0, 0, MPI_COMM_WORLD, &send_handle);
      MPI_Wait(&send_handle, MPI_STATUS_IGNORE);
      MPI_Wait(&recv_handle, MPI_STATUS_IGNORE);
    }

#pragma omp parallel for
    for (std::size_t i = 1; i < ncell - 1; ++i) {
      tmp[i] = data[i] + dt / dx * (numflux(data[i - 1], data[i]) - numflux(data[i], data[i + 1]));
    }

    if (world_rank == 0) {
      std::size_t ix = ncell - 1;
      tmp[ix]
          = data[ix]
          + dt / dx * (numflux(data[ix - 1], data[ix]) - numflux(data[ix], static_cast<double>(halo_recv[0])));
    } else {
      std::size_t ix = 0;
      tmp[ix]
          = data[ix]
          + dt / dx * (numflux(static_cast<double>(halo_recv[nhalo - 1]), data[ix]) - numflux(data[ix], data[ix + 1]));

      ix = ncell - 1;
      tmp[ix]
          = data[ix]
          + dt / dx * (numflux(data[ix - 1], data[ix]) - numflux(data[ix], data[ix]));
    }

#pragma omp parallel for
    for (std::size_t i = 0; i < ncell; ++i) {
      data[i] = tmp[i];
    }

    time += dt;
    ++iter;
    if (world_rank == 0) {
      std::cout << "iter = " << iter << "\ttime = " << time << "\tdt = " << dt << "\n";
    }
  }
  // TODO: clock end here

  auto const final_filename = "data_" + std::to_string(world_rank) + ".final.txt";
  std::ofstream final_file(final_filename);
  for (std::size_t i = 0; i < ncell; ++i) {
    final_file << index_to_x(i, xl, dx) << " " << data[i] << "\n";
  }

  // Finalize the MPI environment.
  MPI_Finalize();
}
