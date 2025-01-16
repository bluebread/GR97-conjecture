#include "intx.hpp"
#include <iostream>

using int256_t = utils::ArbitraryPrecisionInteger<128>;

int main() {
    int256_t a(-1), b(2), c(-3);

    for (int i = 0; i < 1000; ++i) {
        a = a + b;
        std::cout << a.to_string() << std::endl;
    }

    for (int i = 0 ; i < 1000; ++i) {
        a = a + c;
        std::cout << a.to_string() << std::endl;
    }

    return 0; 
}