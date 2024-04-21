import functools as ft
import numpy as np
import scipy as sp
import time

# density matrices for |0> and |1>
ZERO = sp.sparse.coo_array(np.array([[1,0],[0,0]],dtype=np.complex_)) 
ONE = sp.sparse.coo_array(np.array([[0,0],[0,1]],dtype=np.complex_))

# Pauli gates and H
I = sp.sparse.coo_array(np.array([[1,0],[0,1]],dtype=np.complex_))
X = sp.sparse.coo_array(np.array([[0,1],[1,0]],dtype=np.complex_))
Y = sp.sparse.coo_array(np.array([[0,-1j],[1j,0]],dtype=np.complex_))
Z = sp.sparse.coo_array(np.array([[1,0],[0,-1]],dtype=np.complex_))
H = sp.sparse.coo_array(np.array([[1,1],[1,-1]],dtype=np.complex_)/np.sqrt(2))

# perform tensor product on a list of matrices
# returns resulting product
def tensor(matrices):
    return ft.reduce(lambda x,y: sp.sparse.kron(x,y,format='coo'), matrices)

# Z and X stabilizers
# stored in 2D arrays for multiplication optimization reasons
Zstabs=[[tensor([Z,I,Z,I,Z,I,Z]),tensor([I,Z,I,Z,I,Z,I,Z])],
        [tensor([I,Z,Z,I,I,Z,Z]),tensor([I,I,Z,Z,I,I,Z,Z])],
        [tensor([I,I,I,Z,Z,Z,Z]),tensor([I,I,I,I,Z,Z,Z,Z])],
        [tensor([I,I,I,I,I,I,I]),tensor([Z,Z,Z,Z,Z,Z,Z,Z])],
        [tensor([I,I,Z,I,I,I,Z]),tensor([I,I,I,Z,I,I,I,Z])],
        [tensor([I,I,I,I,Z,I,Z]),tensor([I,I,I,I,I,Z,I,Z])],
        [tensor([I,I,I,I,I,Z,Z]),tensor([I,I,I,I,I,I,Z,Z])],
        [tensor([I,I,I,I,I,I,I]),tensor([I,I,Z,Z,I,I,Z,Z])],
        [tensor([I,I,I,I,I,I,I]),tensor([I,I,I,I,Z,Z,Z,Z])],
        [tensor([I,I,I,I,I,I,I]),tensor([I,Z,I,Z,I,Z,I,Z])]]


Xstabs=[[tensor([X,I,X,I,X,I,X]),tensor([I,X,I,X,I,X,I,X])],
        [tensor([I,X,X,I,I,X,X]),tensor([I,I,X,X,I,I,X,X])],
        [tensor([I,I,I,X,X,X,X]),tensor([I,I,I,I,X,X,X,X])],
        [tensor([I,I,I,I,I,I,I]),tensor([X,X,X,X,X,X,X,X])]]

# dictionary for fast log base 2 lookup
# handles 1 to 2^63, with 2^63 being one larger than the maximum possible array dimension
lb={1:0, 2:1, 4:2, 8:3, 16:4, 32:5, 64:6, 128:7, 256:8, 512:9, 1024:10, 2048:11, 4096:12,
    8192:13, 16384:14, 32768:15, 65536:16, 131072:17, 262144:18, 524288:19, 1048576:20,
    2097152:21, 4194304:22, 8388608:23, 16777216:24, 33554432:25, 67108864:26, 134217728:27,
    268435456:28, 536870912:29, 1073741824:30, 2147483648:31, 4294967296:32, 8589934592:33,
    17179869184:34, 34359738368:35, 68719476736:36, 137438953472:37, 274877906944:38,
    549755813888:39, 1099511627776:40, 2199023255552:41, 4398046511104:42, 8796093022208:43,
    17592186044416:44, 35184372088832:45, 70368744177664:46, 140737488355328:47,
    281474976710656:48, 562949953421312:49, 1125899906842624:50, 2251799813685248:51,
    4503599627370496:52, 9007199254740992:53, 18014398509481984:54, 36028797018963968:55,
    72057594037927936:56, 144115188075855872:57, 288230376151711744:58, 576460752303423488:59,
    1152921504606846976:60, 2305843009213693952:61, 4611686018427387904:62, 9223372036854775808:63}

# converts matrix to column vector
# returns column vector
def vec(matrix):
    r,c = matrix.shape
    return matrix.reshape((r*c,1),order='F')

# converts a vector/matrix of any shape to a square matrix
# returns square matrix
# if matrix is not squareable, does approximate square matrix with dimensions n+1 x n
def square_mat(matrix):
    entries = lb[matrix.shape[0]]+lb[matrix.shape[1]]
    return matrix.reshape((2**(entries-entries//2),2**(entries//2)),order='F')

# returns number of qubits in state based on size
def qcount(state):
    r,c = state.shape
    return lb[r]+lb[c]

# applies a list of 15 gates to the state
# returns state after applying gates
def applyGates(state, gates):
    return tensor(gates[7:]) @ state @ tensor(gates[:7]).conj().T

# applies a single-qubit gate to multiple qubits based on input list
# state is a sparse matrix
# gate is the sparse matrix representation of the single-qubit gate
# positions is a list of indices indicating which qubits to apply to
# returns state after applying gates
def applyGateOn(state, gate, positions):
    gates = qcount(state)*[I]
    for i in positions:
        gates[i-1] = gate
    return applyGates(state, gates)

# applies CNOT based on inputs
# state is a sparse matrix
# controls is a list of indices indicating qubits that are controls
# targets is a list of indices indicating qubits that are targets
# returns state after applying CNOts
def CNOT(state, controls, targets):
    CNOT_matrices_zero = qcount(state)*[I]
    CNOT_matrices_one = qcount(state)*[I]
    for i in controls:
        CNOT_matrices_zero[i-1] = ZERO
        CNOT_matrices_one[i-1] = ONE
    for i in targets:
        CNOT_matrices_one[i-1] = X
    return applyGates(state, CNOT_matrices_zero)+applyGates(state, CNOT_matrices_one)
    
# encodes state into [[15,1,3]] Reed-Muller code
# input is a 1-qubit state
# output is a 15-qubit state
def rm_encode(state):
    state = applyGateOn(state, H, [1,2,4,8])
    state = CNOT(state, [1], [3,5,7,9,11,13,15])
    state = CNOT(state, [2], [3,6,7,10,11,14,15])
    state = CNOT(state, [4], [5,6,7,12,13,14,15])
    state = CNOT(state, [8], [9,10,11,12,13,14,15])
    return state
    
# decodes state from [[15,1,3]] Reed-Muller code
# input is a 15-qubit state
# output is a 1-qubit state
def rm_decode(state):
    state = CNOT(state, [8], [9,10,11,12,13,14,15])
    state = CNOT(state, [4], [5,6,7,12,13,14,15])
    state = CNOT(state, [2], [3,6,7,10,11,14,15])
    state = CNOT(state, [1], [3,5,7,9,11,13,15])
    state = CNOT(state, [15], [3,5,6,9,10,12])
    state = applyGateOn(state, H, [1,2,4,8])
    return state

# performs checks for Reed-Muller code
# returns two arrays, one for the measurement results of the Z stabilizers and one for the X stabilizers
def check(state):
    state = square_mat(state)
    Zchecks = np.zeros(10,dtype=np.complex_)
    Xchecks = np.zeros(4,dtype=np.complex_)
    state_c = state.conj()
    for i in range(10):
        Zchecks[i] = (state_c*(Zstabs[i][1] @ state @ Zstabs[i][0])).sum()
    for i in range(4):
        Xchecks[i] = (state_c*(Xstabs[i][1] @ state @ Xstabs[i][0])).sum()
    return Zchecks, Xchecks
    
# Code used for timing
def benchmark(n=20):
    avg = 0
    for i in range(n):
        start_time = time.perf_counter()
        
        rm_state = sp.sparse.coo_array(([1],([0],[0])),shape=(2,1),dtype=np.complex_)
        #rm_state = sp.sparse.coo_array(([1],([0],[0])),shape=(2**8,2**7),dtype=np.complex_)
        rm_state = rm_encode(rm_state)
        for i in range(1,16):
            rm_state = applyGateOn(rm_state,X,[i])
            check(rm_state)
            rm_state = applyGateOn(rm_state,X,[i])
        for i in range(1,16):
            rm_state = applyGateOn(rm_state,Z,[i])
            check(rm_state)
            rm_state = applyGateOn(rm_state,Z,[i])
        rm_state = rm_decode(rm_state)
        end_time = time.perf_counter()-start_time
        print(end_time)
        avg += end_time/n
    print(f"Avg:{avg}")