//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #1: Parallel Bucket Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0

void insertionSort(std::vector<int> &bucket)
{
    /* You may print out the data size in each bucket here to see how severe the load imbalance is */
    for (int i = 1; i < bucket.size(); ++i)
    {
        int key = bucket[i];
        int j = i - 1;

        while (j >= 0 && bucket[j] > key)
        {
            bucket[j + 1] = bucket[j];
            j--;
        }

        bucket[j + 1] = key;
    }
}

/**
 * TODO: Parallel Bucket Sort with MPI (optimized for large datasets)
 * @param vec: input vector for sorting
 * @param num_buckets: number of buckets
 * @param numtasks: number of processes for sorting
 * @param taskid: the rank of the current process
 * @param status: MPI_Status for message passing
 */
void bucketSort(std::vector<int> &vec, int num_buckets, int numtasks, int taskid, MPI_Status *status)
{
    if (vec.empty()) {
        std::cerr << "Error: Input vector is empty. No sorting performed." << std::endl;
        return;
    }

    int max_val = *std::max_element(vec.begin(), vec.end());
    int min_val = *std::min_element(vec.begin(), vec.end());

    int range = max_val - min_val + 1;
    int bucket_size = std::max(1, range / num_buckets); // prevent division by zero

    if (taskid == MASTER)
    {
        int chunk_size = vec.size() / numtasks;
        int remainder = vec.size() % numtasks;

        for (int i = 1; i < numtasks; ++i)
        {
            int start_index = i * chunk_size + std::min(i, remainder);
            int end_index = start_index + chunk_size + (i < remainder ? 1 : 0);
            int send_result = MPI_Send(&vec[start_index], end_index - start_index, MPI_INT, i, 0, MPI_COMM_WORLD);
            if (send_result != MPI_SUCCESS) {
                std::cerr << "Error: MPI_Send failed for process " << i << std::endl;
                MPI_Abort(MPI_COMM_WORLD, send_result);
            }
        }

        int start_index = 0;
        int end_index = chunk_size + (remainder > 0 ? 1 : 0);
        vec = std::vector<int>(vec.begin() + start_index, vec.begin() + end_index);
    }
    else
    {
        MPI_Probe(MASTER, 0, MPI_COMM_WORLD, status);
        int data_size;
        MPI_Get_count(status, MPI_INT, &data_size);
        vec.resize(data_size);
        int recv_result = MPI_Recv(vec.data(), data_size, MPI_INT, MASTER, 0, MPI_COMM_WORLD, status);
        if (recv_result != MPI_SUCCESS) {
            std::cerr << "Error: MPI_Recv failed for process " << taskid << std::endl;
            MPI_Abort(MPI_COMM_WORLD, recv_result);
        }
    }

    std::vector<std::vector<int>> buckets(num_buckets);
    for (int num : vec)
    {
        int index = (num - min_val) / bucket_size;
        index = std::min(index, num_buckets - 1);
        buckets[index].push_back(num);
    }

    for (std::vector<int> &bucket : buckets)
    {
        if (!bucket.empty())
        {
            insertionSort(bucket);
        }
    }

    std::vector<int> local_sorted;
    for (const auto &bucket : buckets)
    {
        local_sorted.insert(local_sorted.end(), bucket.begin(), bucket.end());
    }

    if (taskid == MASTER)
    {
        std::vector<int> final_result;
        final_result.insert(final_result.end(), local_sorted.begin(), local_sorted.end());

        for (int i = 1; i < numtasks; ++i)
        {
            MPI_Probe(i, 1, MPI_COMM_WORLD, status);
            int received_size;
            MPI_Get_count(status, MPI_INT, &received_size);
            std::vector<int> temp(received_size);
            int recv_result = MPI_Recv(temp.data(), received_size, MPI_INT, i, 1, MPI_COMM_WORLD, status);
            if (recv_result != MPI_SUCCESS) {
                std::cerr << "Error: MPI_Recv failed for process " << i << std::endl;
                MPI_Abort(MPI_COMM_WORLD, recv_result);
            }
            final_result.insert(final_result.end(), temp.begin(), temp.end());
        }

        std::sort(final_result.begin(), final_result.end());
        vec = final_result;
    }
    else
    {
        int send_result = MPI_Send(local_sorted.data(), local_sorted.size(), MPI_INT, MASTER, 1, MPI_COMM_WORLD);
        if (send_result != MPI_SUCCESS) {
            std::cerr << "Error: MPI_Send failed for process " << taskid << std::endl;
            MPI_Abort(MPI_COMM_WORLD, send_result);
        }
    }
}


int main(int argc, char **argv)
{
    // Verify input argument format
    if (argc != 4)
    {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable dist_type vector_size bucket_num\n");
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    const DistType dist_type = str_2_dist_type(std::string(argv[1]));
    const int size = atoi(argv[2]);
    const int bucket_num = atoi(argv[3]);
    std::vector<int> vec = genRandomVec(size, dist_type); // use default seed
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    bucketSort(vec, bucket_num, numtasks, taskid, &status);

    if (taskid == MASTER)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                  << std::endl;

        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}