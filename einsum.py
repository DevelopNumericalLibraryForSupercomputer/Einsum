import numpy as np
import itertools
from functools import reduce
import time
import einsum

def reduce_mult(L):
    return reduce(lambda x, y: x*y, L)

seedval = 42
np.random.seed(seedval)


#############################################
## einsum input                           
A = np.random.rand(3,4,8,4)
B = np.random.rand(3,8,9,10)
expr = 'ijka,ikbc->abcj'
#############################################

#############################################
## case 1. einsum implmenetation in python ##
#############################################
inputs = [A,B]
start_time = time.time()

qry_expr, res_expr = expr.split('->')
inputs_expr = qry_expr.split(',')
inputs_expr, res_expr
#(['ij', 'jk'], 'ki')

keys = set([(key, size) for keys, input in zip(inputs_expr, inputs) for key, size in list(zip(keys, input.shape))])
#{('i', 3), ('j', 4), ('k', 2)}

sizes = dict(keys)
#{'i': 3, 'j': 4, 'k': 2}

ranges = [range(size) for _, size in keys]
#[range(0, 2), range(0, 3), range(0, 4)]

to_key = sizes.keys()
#['k', 'i', 'j']  #dict_keys(['k', 'j', 'i']))

domain = itertools.product(*ranges)

res = np.zeros([sizes[key] for key in res_expr])

for indices in domain:
    vals = {k: v for v, k in zip(indices, to_key)}
    #print(vals)
    res_ind = tuple(zip([vals[key] for key in res_expr]))
    inputs_ind = [tuple(zip([vals[key] for key in expr])) for expr in inputs_expr]
    #print(inputs_ind, res_ind)
    res[res_ind] += reduce_mult([M[i] for M, i in zip(inputs, inputs_ind)])
end_time = time.time()
C1 = res
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.6f} seconds")
#############################################
## case 1. end                             ##
#############################################

#############################################
## case 2. numpy einsum                    
start_time2 = time.time()
C2 = np.einsum(expr, A,B)
end_time2 = time.time()
#############################################
execution_time2 = end_time2 - start_time2
print(f"Execution time: {execution_time2:.6f} seconds")

#############################################
## case 3. C++ implementation of einsum    
start_time3 = time.time()
C3 = einsum.c_einsum(expr, A,B)
end_time3 = time.time()
#############################################

execution_time3 = end_time3 - start_time3
print(f"Execution time: {execution_time3:.6f} seconds")

# result verification
print(np.linalg.norm(C1-C2))
print(np.linalg.norm(C3-C2))
