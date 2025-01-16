#ifndef __BINOMIAL_HPP__
#define __BINOMIAL_HPP__

#include "intx.hpp"

namespace utils {
    template<int N>
    class BinomialCoefficient {
    private:
        intx_t dp[(N + 1) * (N + 1)]; // dynamic programming table
    public:
        __host__ BinomialCoefficient() {
            for (int i = 0; i <= N; ++i) {
                dp[i * (N + 1) + 0] = 1;
                dp[i * (N + 1) + i] = 1;
            }
            for (int i = 1; i <= N; ++i) {
                for (int j = 1; j < i; ++j) {
                    dp[i * (N + 1) + j] = dp[(i - 1) * (N + 1) + j - 1] + dp[(i - 1) * (N + 1) + j];
                }
            }
        }
        __host__ intx_t operator()(int n, int k) const {
            return dp[n * (N + 1) + k];
        }
    };
    
};

#endif // __BINOMIAL_HPP__