#include <iostream>
#include "layernorm.h"

int main() {
    fixed_point input[ROWS][COLS] = {
        {0.2, 0.1, 0.3},
        {0.5, 0.1, 0.1}
    };
    
    fixed_point output[ROWS][COLS];

    layernorm(input, output);

    std::cout << "Input matrix:" << std::endl;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            std::cout << static_cast<float>(input[i][j]) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Layer-normalized matrix:" << std::endl;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            std::cout << static_cast<float>(output[i][j]) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
