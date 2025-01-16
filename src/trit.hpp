#ifndef __TRIT_HPP__
#define __TRIT_HPP__

#include <string>

namespace utils {
    enum class Trit {
        Zero = 0b00,
        One = 0b01,
        Two = 0b10,
        Unknown = 0b11,
    };

    template<int num_trits>
    class TritVector {
    private:
        static const int num_blocks = (num_trits * 2 / 64) + ((num_trits * 2) % 64 != 0);
        uint64_t blocks[num_blocks];
    public:
        __host__ __device__ TritVector() {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                blocks[i] = UINT64_MAX;
            }
        }
        __host__ __device__ TritVector(const TritVector& other) {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                blocks[i] = other.blocks[i];
            }
        }
        __host__ __device__ TritVector& operator=(const TritVector& other) {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                blocks[i] = other.blocks[i];
            }
            return *this;
        }
        __host__ __device__ TritVector operator|(const TritVector& other) const {
            TritVector<num_trits> result;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                result.blocks[i] = blocks[i] | other.blocks[i];
            }
            return result;
        }
        __host__ __device__ TritVector operator&(const TritVector& other) const {
            TritVector<num_trits> result;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                result.blocks[i] = blocks[i] & other.blocks[i];
            }
            return result;
        }
        __host__ __device__ TritVector operator^(const TritVector& other) const {
            TritVector<num_trits> result;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                result.blocks[i] = blocks[i] ^ other.blocks[i];
            }
            return result;
        }
        __host__ __device__ TritVector operator~() const {
            TritVector<num_trits> result;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                result.blocks[i] = ~blocks[i];
            }
            return result;
        }
        __host__ __device__ TritVector operator<<(int shift) const {
            TritVector<num_trits> result;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                result.blocks[i] = blocks[i] << (2 * shift);
            }
            return result;
        }
        __host__ __device__ TritVector operator>>(int shift) const {
            TritVector<num_trits> result;
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                result.blocks[i] = blocks[i] >> (2 * shift);
            }
            return result;
        }
        __host__ __device__ TritVector& operator<<=(int shift) {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                blocks[i] <<= (2 * shift);
            }
            return *this;
        }
        __host__ __device__ TritVector& operator>>=(int shift) {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                blocks[i] >>= (2 * shift);
            }
            return *this;
        }
        __host__ __device__ bool operator==(const TritVector& other) const {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                if (blocks[i] != other.blocks[i]) {
                    return false;
                }
            }
            return true;
        }
        __host__ __device__ bool operator!=(const TritVector& other) const {
            return !(*this == other);
        }
        __host__ __device__ void operator|=(const TritVector& other) {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                blocks[i] |= other.blocks[i];
            }
        }
        __host__ __device__ void operator&=(const TritVector& other) {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                blocks[i] &= other.blocks[i];
            }
        }
        __host__ __device__ void operator^=(const TritVector& other) {
            #pragma unroll
            for (int i = 0; i < num_blocks; ++i) {
                blocks[i] ^= other.blocks[i];
            }
        }
        __host__ __device__ Trit operator[](int index) const {
            return static_cast<Trit>((blocks[index / 32] >> (index % 32) * 2) & 0b11);
        }
        __host__ __device__ int size() const {
            return num_trits;
        }
        __host__ __device__ void set(int index, Trit value) {
            blocks[index / 32] &= ~(0b11 << (index % 32) * 2);
            blocks[index / 32] |= static_cast<uint64_t>(value) << (index % 32) * 2;
        }
        __host__ std::string to_string() const {
            std::string result;
            for (int i = num_trits - 1; i >= 0; --i) {
                switch ((*this)[i]) {
                    case 0b00:
                        result += (num_trits % 2) ? "0" : "-";
                        break;
                    case 0b01:
                        result += (num_trits % 2) ? "1" : "0";
                        break;
                    case 0b10:
                        result += (num_trits % 2) ? "2" : "+";
                        break;
                    case 0b11:
                        result += "?";
                        break;
                }
            }
            return result;
        }
    };
};

#endif // __TRIT_HPP__