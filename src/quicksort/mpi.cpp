//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #2: Parallel Quick Sort with K-Way Merge using MPI
//

#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <tuple>

#include <mpi.h>

#include "../utils.hpp"

#define MASTER 0

int partition(std::vector<int> &vec, int low, int high) {
    int pivot = vec[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (vec[j] <= pivot) {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }
    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

/**
 * TODO: Implement parallel quick sort with MPI
 */
void quickSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    int n = vec.size();
    int base_local_n = n / numtasks;
    int remainder = n % numtasks;
    int local_n = (taskid < remainder) ? base_local_n + 1 : base_local_n;
    
    // calculate the offset for each process
    int offset = 0;
    for (int i = 0; i < taskid; ++i) {
        offset += (i < remainder) ? base_local_n + 1 : base_local_n;
    }
    
    std::vector<int> local_data(local_n);
    
    if (taskid == MASTER) {
        // master distributes data to each process
        offset = 0;
        for (int i = 0; i < numtasks; i++) {
            int current_local_n = (i < remainder) ? base_local_n + 1 : base_local_n;
            if (i == MASTER) {
                local_data.assign(vec.begin(), vec.begin() + current_local_n);
            } else {
                MPI_Send(vec.data() + offset, current_local_n, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
            offset += current_local_n;
        }
    } else {
        // worker processes receive their partition of data
        MPI_Recv(local_data.data(), local_n, MPI_INT, MASTER, 0, MPI_COMM_WORLD, status);
    }
    
    std::sort(local_data.begin(), local_data.end());
    
    // gather back
    if (taskid == MASTER) {
        std::vector<int> sorted_data(n);
        offset = 0;
        std::copy(local_data.begin(), local_data.end(), sorted_data.begin() + offset);
        offset += local_n;
        
        for (int i = 1; i < numtasks; i++) {
            int current_local_n = (i < remainder) ? base_local_n + 1 : base_local_n;
            MPI_Recv(sorted_data.data() + offset, current_local_n, MPI_INT, i, 0, MPI_COMM_WORLD, status);
            offset += current_local_n;
        }
        
        auto cmp = [](const std::tuple<int, int, int>& a, const std::tuple<int, int, int>& b) {
            return std::get<0>(a) > std::get<0>(b);
        };
        std::priority_queue<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>, decltype(cmp)> min_heap(cmp);
        
        std::vector<int> indices(numtasks, 0);
        offset = 0;
        for (int i = 0; i < numtasks; ++i) {
            int current_local_n = (i < remainder) ? base_local_n + 1 : base_local_n;
            if (current_local_n > 0 && indices[i] < current_local_n) {
                min_heap.emplace(sorted_data[offset], i, offset);
            }
            offset += current_local_n;
        }
        
        vec.clear();
        while (!min_heap.empty()) {
            auto top = min_heap.top();
            int value = std::get<0>(top);
            int proc_id = std::get<1>(top);
            int idx = std::get<2>(top);
            min_heap.pop();
            vec.push_back(value);
            indices[proc_id]++;
            
            int current_local_n = (proc_id < remainder) ? base_local_n + 1 : base_local_n;
            if (indices[proc_id] < current_local_n) {
                min_heap.emplace(sorted_data[idx + 1], proc_id, idx + 1);
            }
        }
    } else {
        MPI_Send(local_data.data(), local_n, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
    }
}



int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable dist_type vector_size\n"
            );
    }
    const DistType dist_type = str_2_dist_type(std::string(argv[1]));
    const int size = atoi(argv[2]);
    std::vector<int> vec = genRandomVec(size, dist_type); // use default seed
    std::vector<int> vec_clone = vec;

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

    auto start_time = std::chrono::high_resolution_clock::now();
    
    quickSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Quick Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}
