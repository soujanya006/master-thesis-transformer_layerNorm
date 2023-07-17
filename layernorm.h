#include <ap_fixed.h>
#include <cmath>

typedef ap_fixed<32, 16> fixed_point;

constexpr int ROWS = 2;
constexpr int COLS = 3;

void layernorm(fixed_point input[ROWS][COLS], fixed_point output[ROWS][COLS]);
