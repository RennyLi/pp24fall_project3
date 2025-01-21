// Created by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #3: Parallel Sorting with Regular Sampling using MPI
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <mpi.h>
#include <algorithm>
#include <vector>

#include "../utils.hpp"

#define MASTER 0

/**
 * The following are variables for each process
 * Feel free to delete them or create new ones 
 */
std::vector<int> DATA;                 // Input dataset for sorting
std::vector<int> SortedData;           // Output sorted data
std::vector<int> localData;            // Data that each process gets
std::vector<int> localRegularSamples;  // Regular samples of each process
std::vector<int> regularSamples;       // Global regular samples in master
std::vector<int> pivots;               // Pivots for load balancing
std::vector<int> splitters;            // splitting indices
std::vector<int> mergedArray;          // locally merged array
std::vector<int> lengths;              // lengths of splitted array pieces
std::vector<int> obtainedKeys;         // obtained keys from other processors

DistType DIST_TYPE;          // Distribution type of the input dataset
int obtainedKeysSize = 0;    // data size obtained from other processes
int dataCurrentProc;         // Data Size the Current Process Gets
int T;                       // Total number of processes for sorting
int SIZE;                    // Size of the dataset to sort
int rank;                    // Rank of the current process

// data distribution phase
void phase_0()
{
    // regular data size that a processor will get
    int dataPerProc = std::ceil((double)SIZE / T);
    // the actual data size that the processor will get
    dataCurrentProc = (rank == T - 1) ? SIZE - (T - 1) * dataPerProc : dataPerProc;
    // allocate sufficient memory for the local array
    localData = std::vector<int>(dataCurrentProc, 0);
    std::vector<int> lenEachProc(T, 0); // Array size for each process
    std::vector<int> displacements(T);  // Displacement index for each proc
    if (rank == MASTER)
    {
        DATA = genRandomVec(SIZE, DIST_TYPE); // use default seed
        for (int index = 0; index < T; index++)
        {
            lenEachProc[index] = (index == T - 1) ? SIZE - (T - 1) * dataPerProc : dataPerProc;
        }
        displacements = prefixSum(lenEachProc);
    }
    // Scatter dataset to each processor
    MPI_Scatterv(DATA.data(), lenEachProc.data(), displacements.data(), MPI_INT, localData.data(), dataCurrentProc, MPI_INT, MASTER, MPI_COMM_WORLD);
}

/**
 * TODO: local sorting and regular sampling phase
 * 1. You need to sort the local partition by any algorithm you want
 * 2. You need to pick T local regular samples
 */
void phase_1()
{
    std::sort(localData.begin(), localData.end());

    localRegularSamples.resize(T);
    for (int i = 0; i < T; i++)
    {
        int index = i * dataCurrentProc / T;
        if (index >= dataCurrentProc)
            index = dataCurrentProc - 1;
        localRegularSamples[i] = localData[index];
    }
}

/**
 * TODO: Pivot Selection Phase
 * 1. Gather all the local samples from each process
 * 2. Select (T - 1) global pivots
 */
void phase_2()
{
    // gather all local regular samples to master process
    if (rank == MASTER)
    {
        regularSamples.resize(T * T);
    }
    MPI_Gather(localRegularSamples.data(), T, MPI_INT, regularSamples.data(), T, MPI_INT, MASTER, MPI_COMM_WORLD);

    // select T-1 global pivots from gathered regular samples
    if (rank == MASTER)
    {
        std::sort(regularSamples.begin(), regularSamples.end());
        pivots.resize(T - 1);
        for (int i = 1; i < T; i++)
        {
            pivots[i - 1] = regularSamples[i * T];
        }
    }

    // broadcast pivots to all processes
    pivots.resize(T - 1); 
    MPI_Bcast(pivots.data(), T - 1, MPI_INT, MASTER, MPI_COMM_WORLD);
}

/**
 * TODO: Split the data pieces and exchange them across processes
 */
void phase_3()
{
    splitters.resize(T + 1);
    splitters[0] = 0;
    splitters[T] = dataCurrentProc;
    int pivotIndex = 0;
    for (int i = 0; i < dataCurrentProc; i++)
    {
        if (pivotIndex < pivots.size() && localData[i] > pivots[pivotIndex])
        {
            splitters[++pivotIndex] = i;
        }
    }
    while (pivotIndex < T - 1)
    {
        splitters[++pivotIndex] = dataCurrentProc;
    }

    lengths.resize(T);
    for (int i = 0; i < T; i++)
    {
        lengths[i] = splitters[i + 1] - splitters[i];
    }

    std::vector<int> recvLengths(T);
    MPI_Alltoall(lengths.data(), 1, MPI_INT, recvLengths.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int totalRecvSize = std::accumulate(recvLengths.begin(), recvLengths.end(), 0);
    obtainedKeys.resize(totalRecvSize);
    std::vector<int> sendDispls = prefixSum(lengths);
    std::vector<int> recvDispls = prefixSum(recvLengths);

    // exchange data
    MPI_Alltoallv(localData.data(), lengths.data(), sendDispls.data(), MPI_INT,
                  obtainedKeys.data(), recvLengths.data(), recvDispls.data(), MPI_INT, MPI_COMM_WORLD);
}

/**
 * TODO: Merge local partitions
 * You can use k-way merge in Task #2 if you want
 */
void phase_4()
{
    std::sort(obtainedKeys.begin(), obtainedKeys.end());
    mergedArray = obtainedKeys;
}

/**
 * TODO: Merge all local arrays into SortedData in master process
 */
void phase_merge()
{
    std::vector<int> recvCounts(T);
    int mergedSize = mergedArray.size();
    MPI_Gather(&mergedSize, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    std::vector<int> displs = prefixSum(recvCounts);
    if (rank == MASTER)
    {
        SortedData.resize(SIZE);
    }

    MPI_Gatherv(mergedArray.data(), mergedSize, MPI_INT, SortedData.data(), recvCounts.data(), displs.data(), MPI_INT, MASTER, MPI_COMM_WORLD);
}

/**
 * You can measure the time of each phase with this function
 */
void measureTime(void (*fun)(), char *processorName, char *title, int shouldLog)
{
    if (shouldLog)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        fun();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        printf("[%s:%d] %s took %ld ms\n", processorName, rank, title, elapsed_time.count());
    }
    else
    {
        fun();
    }
}

int main(int argc, char *argv[])
{
    // Verify input argument format
    if (argc != 3)
    {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable dist_type vector_size\n");
    }
    DIST_TYPE = str_2_dist_type(std::string(argv[1]));
    SIZE = atoi(argv[2]); // data size to sort

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &T);      // how many processors are available
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // what's my rank?
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len); // What's processor name

    // Phase 0: Data distribution
    measureTime(phase_0, processor_name, "Phase 0", rank == 0);
    MPI_Barrier(MPI_COMM_WORLD);

    auto start_time = std::chrono::high_resolution_clock::now();

    // PHASE 1
    // For now, measureTime function does not print the time consumption for each phase, change the last param to 1 if you want to print it out
    measureTime(phase_1, processor_name, "Phase 1", 0);
    // PHASE 2
    measureTime(phase_2, processor_name, "Phase 2", 0);
    // PHASE 3
    measureTime(phase_3, processor_name, "Phase 3", 0);
    // PHASE 4
    measureTime(phase_4, processor_name, "Phase 4", 0);

    MPI_Barrier(MPI_COMM_WORLD);

    // PHASE Merge
    measureTime(phase_merge, processor_name, "Phase Merge", rank == 0);

    if (rank == MASTER)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        std::cout << "Sorting Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " \t\tmilliseconds" << std::endl;
        checkSortResult(DATA, SortedData); // check if sorted
    }

    MPI_Finalize();
    return 0;
}
