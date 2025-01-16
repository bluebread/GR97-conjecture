#include <iostream>
#include <vector>
#include <iterator>
#include <omp.h>

#include "config.hpp"
#include "utils.hpp"

__host__ void print_info() {
    #ifdef CPU_ONLY
        std::cout << "Running on CPU" << std::endl;
        std::cout << "Number of threads: " << omp_get_max_threads() << std::endl;
        std::cout << "Number of cores: " << omp_get_num_procs() << std::endl;
        std::cout << "Number of threads per core: " << omp_get_max_threads() / omp_get_num_procs() << std::endl;
        std::cout << "Number of threads per socket: " << omp_get_max_threads() / omp_get_num_procs() << std::endl;
        std::cout << "Number of threads per GPU: " << 0 << std::endl;
        std::cout << "Number of GPUs: " << 0 << std::endl;
    #else

    #endif
}

__host__ void recursive_search(
    std::vector<utils::TritVector<FOLDED_VECTOR_LENGTH>> &solutions,
    const utils::BinomialCoefficient<NUM_VARIABLES> &binom,
    utils::TritVector<FOLDED_VECTOR_LENGTH> &vector,
    int depth,
    utils::intx_t suffix_sum
) {
    if (depth == FOLDED_VECTOR_LENGTH) {
        if (suffix_sum == 0) {
            solutions.push_back(vector);
        }
        return;
    }
    else if (depth == SPAWN_DEPTH) {
        int idx = FOLDED_VECTOR_LENGTH - depth - 1;
        utils::intx_t coeff = binom(NUM_VARIABLES, depth);
        std::vector<utils::TritVector<FOLDED_VECTOR_LENGTH>> slave1_solutions;
        std::vector<utils::TritVector<FOLDED_VECTOR_LENGTH>> slave2_solutions;

        if (NUM_VARIABLES % 2 == 0) {
            #pragma omp task 
            {
                utils::TritVector<FOLDED_VECTOR_LENGTH> slave1_vector = vector;

                slave1_vector.set(idx, utils::Trit::Zero);
                recursive_search(slave1_solutions, binom, slave1_vector, depth + 1, suffix_sum);
            }

            #pragma omp task 
            {
                utils::TritVector<FOLDED_VECTOR_LENGTH> slave2_vector = vector;

                slave2_vector.set(idx, utils::Trit::One);
                recursive_search(slave2_solutions, binom, slave2_vector, depth + 1, suffix_sum + coeff);
            }

            vector.set(idx, utils::Trit::Two);
            recursive_search(solutions, binom, vector, depth + 1, suffix_sum + (coeff << 1));
        }
        else {
            #pragma omp task
            {
                utils::TritVector<FOLDED_VECTOR_LENGTH> slave1_vector = vector;

                slave1_vector.set(idx, utils::Trit::Zero);
                recursive_search(slave1_solutions, binom, slave1_vector, depth + 1, suffix_sum - coeff);
            }

            #pragma omp task
            {
                utils::TritVector<FOLDED_VECTOR_LENGTH> slave2_vector = vector;

                slave2_vector.set(idx, utils::Trit::One);
                recursive_search(slave2_solutions, binom, slave2_vector, depth + 1, suffix_sum);
            }

            vector.set(idx, utils::Trit::Two);
            recursive_search(solutions, binom, vector, depth + 1, suffix_sum + coeff);
        }

        #pragma omp taskwait

        solutions.insert(
            solutions.end(), 
            std::make_move_iterator(slave1_solutions.begin()), 
            std::make_move_iterator(slave1_solutions.end())
        );
        solutions.insert(
            solutions.end(), 
            std::make_move_iterator(slave2_solutions.begin()), 
            std::make_move_iterator(slave2_solutions.end())
        );

        return;
    }
    else {
        int idx = FOLDED_VECTOR_LENGTH - depth - 1;
        utils::intx_t coeff = binom(NUM_VARIABLES, depth);

        if (NUM_VARIABLES % 2 == 0) {
            vector.set(idx, utils::Trit::Zero);
            recursive_search(solutions, binom, vector, depth + 1, suffix_sum);
            vector.set(idx, utils::Trit::One);
            recursive_search(solutions, binom, vector, depth + 1, suffix_sum + coeff);
            vector.set(idx, utils::Trit::Two);
            recursive_search(solutions, binom, vector, depth + 1, suffix_sum + (coeff << 1));
        }
        else {
            vector.set(idx, utils::Trit::Zero);
            recursive_search(solutions, binom, vector, depth + 1, suffix_sum - coeff);
            vector.set(idx, utils::Trit::One);
            recursive_search(solutions, binom, vector, depth + 1, suffix_sum);
            vector.set(idx, utils::Trit::Two);
            recursive_search(solutions, binom, vector, depth + 1, suffix_sum + coeff);
        }

        return;
    }
}

__host__ void validate_conjecture() {
    #pragma omp parallel
    {
        #pragma omp single
        {
            utils::BinomialCoefficient<NUM_VARIABLES> binom;
            utils::TritVector<FOLDED_VECTOR_LENGTH> folded_vector;
            std::vector<utils::TritVector<FOLDED_VECTOR_LENGTH>> solutions;

            recursive_search(solutions, binom, folded_vector, 0, 0);

            for (const auto &solution : solutions) {
                std::cout << solution.to_string() << std::endl;
            }
        }
    }
}

__host__ int main() {
    print_info();

    return 0;
}