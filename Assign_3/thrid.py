from mpi4py import MPI
import numpy as np

N = 10  # total number of elements
n = 4  # number of processors

# create the array and initialize it with sequential values
arr = np.arange(1, N + 1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != n:
    print(f"Error: must run with {n} processes")
    comm.Abort(1)

# calculate the local sum
local_sum = np.sum(arr[rank * N // size:(rank + 1) * N // size])

# reduce the local sums to get the global sum
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

# send the local sum to process 0
if rank != 0:
    comm.send(local_sum, dest=0)
else:
    # process 0 receives the local sums and prints the intermediate and final results
    print(f"Rank {rank} local sum: {local_sum}")
    for i in range(1, size):
        local_sum = comm.recv(source=i)
        print(f"Rank {i} local sum: {local_sum}")
    print(f"Global sum: {global_sum}")

MPI.Finalize()
