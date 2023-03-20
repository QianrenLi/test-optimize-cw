#include <stdio.h>
#include <math.h>
#include "utils.h"


extern void initial_mcs(double *);
extern double run_with_throttle(double const *);

static const int num = 4;

const double epsilon_M = 0.5;
const int e_I = 1;
const int h_K = 1;
const int alpha = 5;

static void vec_scalar_inc(double* vec_dst, double *vec_src, double val) {
    int i, size;
    size = sizeof(vec_src) / sizeof(double);
    for (i=0; i<size; i++) {
        vec_dst[i] = vec_src[i] + val;
    }
}

void gradient_search() {
    int counter;
    int _direction, _h_fd;
    double _delta_x;

    double x[num] = {0}, x_tmp[num] = {0}, x_his[num] = {0};
    double qos, qos_tmp;

    initial_mcs(x);

    counter = 0;
    while (1) {
        counter ++;
        _direction = counter % num;
        _h_fd = max( 1, abs(x[_direction]) ) * epsilon_M;
        vec_scalar_inc(x_tmp, x, _h_fd * e_I);

        qos = run_with_throttle(x);
        qos_tmp = run_with_throttle(x_tmp);

        _delta_x = alpha * h_K * (qos_tmp - qos) / _h_fd;
        vec_scalar_inc( x, x, -_delta_x );
    }
}


static void main(char **args) {
    gradient_search();
}
