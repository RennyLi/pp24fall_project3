//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Modified by Liu Yuxuan on 2024/10/26
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #4: Parallel Merge Sort with OpenMP
//

#include <iostream>
#include <vector>
#include <omp.h>
#include "../utils.hpp"

/**
 * Implement parallel merge algorithm
 */
void merge(std::vector<int>& vec, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    std::vector<int> L(n1);
    std::vector<int> R(n2);

    // copy data to temporary vectors
    for (int i = 0; i < n1; i++) {
        L[i] = vec[l + i];
    }
    for (int i = 0; i < n2; i++) {
        R[i] = vec[m + 1 + i];
    }

    int i = 0; 
    int j = 0; 
    int k = l; 

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            vec[k] = L[i];
            i++;
        } else {
            vec[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        vec[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        vec[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(std::vector<int>& vec, int l, int r, int thread_num) {
    if (l < r) {
        int m = l + (r - l) / 2;

        #pragma omp parallel num_threads(thread_num)
        {
            #pragma omp single
            {
                #pragma omp task
                mergeSort(vec, l, m, thread_num);

                #pragma omp task
                mergeSort(vec, m + 1, r, thread_num);

                #pragma omp taskwait
                merge(vec, l, m, r);
            }
        }
    }
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable dist_type threads_num vector_size\n"
            );
    }
    const DistType dist_type = str_2_dist_type(std::string(argv[1]));
    const int thread_num = atoi(argv[2]);
    const int size = atoi(argv[3]);
    std::vector<int> vec = genRandomVec(size, dist_type); // use default seed
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    mergeSort(vec, 0, size - 1, thread_num);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Merge Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds" << std::endl;

    checkSortResult(vec_clone, vec);
    return 0;
} 
