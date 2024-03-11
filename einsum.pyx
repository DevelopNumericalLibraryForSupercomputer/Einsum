#distutils: language = c++
import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
np.import_array()

# Declare the external C++ function
cdef extern from "einsum_vector.hpp":
    void input_parser(const string input_str, const vector[size_t] a_size, const vector[size_t] b_size,
                      string* result_index, vector[string]* inputs_exprs, map[string, size_t]* total_iter_sizes)
    vector[size_t] calculate_return_size(const string result_index, const map[string, size_t] total_iter_sizes)
    void einsum_core(string result_index, vector[string] inputs_exprs, map[string, size_t] total_iter_sizes,
                 const double* a, const double* b, double* c, const vector[size_t] a_size, const vector[size_t] b_size, vector[size_t] c_size)

def c_einsum(str input_string, np.ndarray A, np.ndarray B):
    cdef vector[size_t] a_size
    cdef vector[size_t] b_size
    cdef vector[size_t] c_size

    for dim in range(A.ndim):
        a_size.push_back(<size_t>(A.shape[dim]))

    for dim in range(B.ndim):
        b_size.push_back(<size_t>(B.shape[dim]))
        
    cdef string result_index
    cdef vector[string] inputs_exprs
    cdef map[string, size_t] total_iter_sizes

    #str ty bytes
    input_parser( <string>input_string.encode('utf-8'), a_size, b_size, &result_index, &inputs_exprs, &total_iter_sizes)

    #get size of c
    c_size = calculate_return_size(result_index, total_iter_sizes)

    #Initialization
    new_C = np.zeros(list(c_size), dtype=np.float64)
    
    # Get direct access to the underlying NumPy array data
    cdef double[::1] A_view = np.ravel(A, order='C')  # Use C-contiguous view
    cdef double[::1] B_view = np.ravel(B, order='C')
    cdef double[::1] C_view = np.ravel(new_C, order='C')

    # summation
    einsum_core(result_index, inputs_exprs, total_iter_sizes, &A_view[0], &B_view[0], &C_view[0], a_size, b_size, c_size)
    
    return new_C
    