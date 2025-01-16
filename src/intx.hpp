#ifndef __UTIL_CU__
#define __UTIL_CU__

#include <string>
#include <cstdint>
#include <limits>

#include "config.hpp"

namespace utils {
    template<int num_bits>
    class ArbitraryPrecisionInteger {
    private:
        static const int num_blocks = (num_bits / 64) + (num_bits % 64 != 0);
        uint64_t blocks[num_blocks];
    public:
        __host__ __device__ ArbitraryPrecisionInteger() {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                blocks[i] = 0;
            }
        }
        __host__ __device__ ArbitraryPrecisionInteger(long value) {
            uint64_t init_v = (value < 0) ? __UINT64_MAX__ : 0;
            #pragma unroll
            for (int i = 1; i < num_blocks; ++i) {
                blocks[i] = init_v;
            }
            blocks[0] = value;
        }
        __host__ __device__ ArbitraryPrecisionInteger(const ArbitraryPrecisionInteger& other) {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                blocks[i] = other.blocks[i];
            }
        }
        __host__ __device__ ArbitraryPrecisionInteger& operator=(const ArbitraryPrecisionInteger& other) {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                blocks[i] = other.blocks[i];
            }
            return *this;
        }
        __host__ __device__ ArbitraryPrecisionInteger operator+(const ArbitraryPrecisionInteger& other) const {
            ArbitraryPrecisionInteger<num_bits> result;
            uint64_t carry = 0;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                uint64_t sum = blocks[i] + other.blocks[i] + carry;
                
                if (sum < blocks[i] || (carry && sum == blocks[i])) {
                    carry = 1;
                } else {
                    carry = 0;
                }
                result.blocks[i] = sum;
            }
            return result;
        }
        __host__ __device__ ArbitraryPrecisionInteger operator-(const ArbitraryPrecisionInteger& other) const {
            ArbitraryPrecisionInteger<num_bits> result;
            uint64_t borrow = 0;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                uint64_t diff = blocks[i] - other.blocks[i] - borrow;
                if (diff > blocks[i] || (borrow && diff == blocks[i])) {
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                result.blocks[i] = diff;
            }
            return result;
        }
        __host__ __device__ bool operator==(const ArbitraryPrecisionInteger& other) const {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                if (blocks[i] != other.blocks[i]) {
                    return false;
                }
            }
            return true;
        }
        __host__ __device__ bool operator!=(const ArbitraryPrecisionInteger& other) const {
            return !(*this == other);
        }
        __host__ __device__ bool operator<(const ArbitraryPrecisionInteger& other) const {
            #pragma unroll
            for (int i = num_blocks - 1; i >= 0; --i) {
                if (blocks[i] < other.blocks[i]) {
                    return true;
                } else if (blocks[i] > other.blocks[i]) {
                    return false;
                }
            }
            return false;
        }
        __host__ __device__ bool operator>(const ArbitraryPrecisionInteger& other) const {
            return other < *this;
        }
        __host__ __device__ bool operator<=(const ArbitraryPrecisionInteger& other) const {
            return !(*this > other);
        }
        __host__ __device__ bool operator>=(const ArbitraryPrecisionInteger& other) const {
            return !(*this < other);
        }
        __host__ __device__ ArbitraryPrecisionInteger operator~() const {
            ArbitraryPrecisionInteger<num_bits> result;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                result.blocks[i] = ~blocks[i];
            }
            return result;
        }
        __host__ __device__ ArbitraryPrecisionInteger operator&(const ArbitraryPrecisionInteger& other) const {
            ArbitraryPrecisionInteger<num_bits> result;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                result.blocks[i] = blocks[i] & other.blocks[i];
            }
            return result;
        }
        __host__ __device__ ArbitraryPrecisionInteger operator|(const ArbitraryPrecisionInteger& other) const {
            ArbitraryPrecisionInteger<num_bits> result;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                result.blocks[i] = blocks[i] | other.blocks[i];
            }
            return result;
        }
        __host__ __device__ ArbitraryPrecisionInteger operator^(const ArbitraryPrecisionInteger& other) const {
            ArbitraryPrecisionInteger<num_bits> result;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                result.blocks[i] = blocks[i] ^ other.blocks[i];
            }
            return result;
        } 
        __host__ __device__ ArbitraryPrecisionInteger operator<<(int shift) const {
            ArbitraryPrecisionInteger<num_bits> result;
            int block_shift = shift / 64;
            int bit_shift = shift % 64;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                if (i + block_shift < num_blocks) {
                    result.blocks[i + block_shift] |= blocks[i] << bit_shift;
                }
                if (bit_shift > 0 && i + block_shift + 1 < num_blocks) {
                    result.blocks[i + block_shift + 1] |= blocks[i] >> (64 - bit_shift);
                }
            }
            return result;
        }
        __host__ __device__ ArbitraryPrecisionInteger operator>>(int shift) const {
            ArbitraryPrecisionInteger<num_bits> result;
            int block_shift = shift / 64;
            int bit_shift = shift % 64;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                if (i - block_shift >= 0) {
                    result.blocks[i - block_shift] |= blocks[i] >> bit_shift;
                }
                if (bit_shift > 0 && i - block_shift - 1 >= 0) {
                    result.blocks[i - block_shift - 1] |= blocks[i] << (64 - bit_shift);
                }
            }
            return result;
        }
        __host__ __device__ ArbitraryPrecisionInteger& operator<<=(int shift) {
            *this = *this << shift;
            return *this;
        }
        __host__ __device__ ArbitraryPrecisionInteger& operator>>=(int shift) {
            *this = *this >> shift;
            return *this;
        }
        __host__ __device__ ArbitraryPrecisionInteger& operator&=(const ArbitraryPrecisionInteger& other) {
            *this = *this & other;
            return *this;
        }
        __host__ __device__ ArbitraryPrecisionInteger& operator|=(const ArbitraryPrecisionInteger& other) {
            *this = *this | other;
            return *this;
        }
        __host__ __device__ ArbitraryPrecisionInteger& operator^=(const ArbitraryPrecisionInteger& other) {
            *this = *this ^ other;
            return *this;
        }
        __host__  std::string to_string() const {
            char buffer[2 + 2 * num_blocks * sizeof(uint64_t) + 1] = "0x";
            char * ptr = buffer + 2;

            for (int i = num_blocks - 1; i >= 0; --i) {
                ptr += std::sprintf(ptr, "%016lx", blocks[i]);
            }

            *ptr = '\0';

            return std::string(buffer);
        }
    };

    using intx_t = ArbitraryPrecisionInteger<NUM_VARIABLES>;
};


#endif