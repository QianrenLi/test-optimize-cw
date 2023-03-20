#ifndef __TAP_UTILS_H__
#define __TAP_UTILS_H__

/****************** Native Part ******************/
// enum SERVICE_TYPE {
//     FILE, //AC2, large file service
//     REAL, //AC2, delay-sensitive service
//     PROJ  //AC1, screen projection service
// } SERVICE_TYPE_T;

typedef struct service {
    const char* name;
    // SERVICE_TYPE type;
    double throttle;//Mbps
    double throughput;//Mbps
    double latency;//ms
} service_t;

typedef struct data {
    int len;
    double total_throughput;
    service_t *head; 
} data_t;

void gradient_search(void);

/****************** Python Part ******************/
#define PY_SIZE_T_CLEAN
#include <Python.h>

void initial_mcs(double *);
double run_with_throttle(double const *);

#endif