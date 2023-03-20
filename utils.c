#include "utils.h"

static PyObject* mod_opt = NULL;
static PyObject* fn_initial_mcs = NULL;
static PyObject* fn_run_with_throttle = NULL;

void init() {
    mod_opt = PyImport_Import( PyString_FromString((char *)"optimize") );
    fn_initial_mcs = PyObject_GetAttrString(mod_opt, (char *)"initial_mcs");
    fn_run_with_throttle = PyObject_GetAttrString(mod_opt, (char *)"run_with_throttle");
}

void initial_mcs(double *dst) {
    int i;
    PyObject *results;

    // call "initial_mcs" function from `optimize.py`
    results = PyObject_CallObject(fn_initial_mcs, NULL);

    // update the return results
    for (i=0; i<sizeof(dst)/sizeof(double); i++) {
        dst[i] = PyFloat_AsDouble(PyList_GetItem(results, i));
    }
}

double run_with_throttle(double const *src) {
    int i, _length;
    PyObject *args;

    // build PyList from src
    _length = sizeof(src) / sizeof(double);
    args = PyList_New(_length);
    for (i=0; i<_length; i++) {
        PyList_SetItem(args, i, PyFloat_FromDouble(src[i]));
    }

    // return QoS result
    return PyFloat_AsDouble(PyObject_CallObject(fn_run_with_throttle, args));
}
