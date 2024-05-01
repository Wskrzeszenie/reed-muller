import functools as ft
import multiprocessing as mp
import numpy as np
import os.path
import scipy as sp
import time

# for the scipy library, COO is faster for general storage, reshaping and tensor products
#                        CSR/CSC is faster for arithmetic

# state kets for |0> and |1>, and unnormalized |+>
ket0 = sp.sparse.coo_array(([1],([0],[0])),shape=(2,1),dtype=np.complex_)
ket1 = sp.sparse.coo_array(([1],([1],[0])),shape=(2,1),dtype=np.complex_)
ket01 = sp.sparse.coo_array(([1,1],([0,1],[0,0])),shape=(2,1),dtype=np.complex_)

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
# returns resulting tensor product

def tensor(matrices):
    return (ft.reduce(lambda x,y: sp.sparse.kron(x,y,format='coo'), matrices)).tocsc()

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

# pre-computed matrices for state encoding
stab_1=[[tensor([ZERO]+6*[I]),tensor(8*[I])],
        [tensor([ONE]+3*[I,X]),tensor(4*[I,X])]]
stab_2=[[tensor([I,ZERO,I,I,I,I,I]),tensor(8*[I])],
        [tensor([I,ONE,X,I,I,X,X]),tensor(2*[I,I,X,X])]]
stab_3=[[tensor([I,I,I,ZERO,I,I,I]),tensor(8*[I])],
        [tensor([I,I,I,ONE,X,X,X]),tensor(4*[I]+4*[X])]]
stab_4=[tensor(7*[I]),tensor([ZERO]+7*[I])+tensor([ONE]+7*[X])]

decode_right = tensor([H @ ket01, H @ ket01, ket01, H @ ket01, ket01, ket01, ket01])
decode_left = tensor([ket01.T @ H, ket01.T, ket01.T, ket01.T, ket01.T, ket01.T, ket01.T, I])

flip_5=[[tensor(7*[I]),tensor(7*[I]+[ZERO])],
        [tensor([I,I,X,I,X,X,I]),tensor([I,X,X,I,X,I,I,ONE])]]

# returns the number of bits between the leftmost 1 and the rightmost 1 for an integer
def bit_distance(num):
    num = bin(num)
    num = num.lstrip('-0b')
    num = num.rstrip('0')
    return len(num)

# converts array/list of qubit positions to bit array used for functions
# positions is an array/list of positions
# returns a number whose bits represent the positions, with rightmost bit representing position 1
def array_to_bits(positions):
    bits = 0
    for i in positions:
        bits |= 1<<(i-1)
    return bits

# converts bit array to array/list of qubit positions
# num is a number whose bits represent the positions, with rightmost bit representing position 1
# optional input indicating format (array by default)
# returns an array/list of positions
def bits_to_array(num, format='array'):
    i = 1
    positions = []
    while num:
        if num&1:
            positions.append(i)
        num >>= 1
        i += 1
    if format=='array':
        return np.array(positions)
    else:
        return positions

# converts matrix to column vector
# returns column vector
def vec(matrix):
    if matrix.shape[1] == 1:
        return matrix
    r,c = matrix.shape
    return matrix.reshape((r*c,1),order='F')

# converts a vector/matrix of any shape to a square matrix
# returns square matrix
# if matrix is not square-able, does approximate square matrix with dimensions (2n,n)
def square_mat(matrix):
    if matrix.shape[0] == matrix.shape[1] or matrix.shape[0] == 2*matrix.shape[1]:
        return matrix
    entries = qcount(matrix)
    return matrix.reshape((1<<(entries-entries//2),1<<(entries//2)),order='F')

# counts number of qubits represented by sparse matrix
# state is an n-qubit state represented as a sparse matrix
# returns number of qubits in state based on size
def qcount(state):
    return int(np.log2(np.prod(state.shape)))

# applies a list of 15 gates to the state
# state is an n-qubit state represented as a sparse matrix
# returns state after applying gates
def apply_gates(state, gates):
    return (tensor(gates[len(gates)//2:]) @ state @ tensor(gates[:len(gates)//2]).conj().T.tocsc())

# applies a single-qubit gate to multiple qubits based on input bit array
# state is an n-qubit state represented as a sparse matrix
# gate is the sparse matrix representation of the single-qubit gate
# positions is a bit array indicating which qubits to apply to
# returns state after applying gates
def apply_gate_on(state, gate, positions):
    state_size = qcount(state)
    gates = state_size*[I]
    for i in range(state_size):
        if positions&(1<<i):
            gates[i] = gate
    return apply_gates(state, gates)

# applies CNOT based on inputs
# state is an n-qubit state represented as a sparse matrix
# controls is a bit array of indices indicating qubits that are controls
# targets is a bit array of indices indicating qubits that are targets
# returns state after applying CNOTs
def CNOT(state, controls, targets):
    state_size = qcount(state)
    CNOT_matrices_zero = state_size*[I]
    CNOT_matrices_one = state_size*[I]
    for i in range(state_size):
        if controls&(1<<i):
            CNOT_matrices_zero[i] = ZERO
            CNOT_matrices_one[i] = ONE
        if targets&(1<<i):
            CNOT_matrices_one[i] = X
    return apply_gates(state, CNOT_matrices_zero)+apply_gates(state, CNOT_matrices_one)
    
# performs partial trace with respect to input bit array
# wrt is a bit array indicating which qubits to trace with respect to
# state is an n-qubit state represented as a sparse matrix
# returns traced state
# assumes state is NOT mixed or has entanglement crossing between the two subsystems
def partial_trace(state, wrt):
    state_size = qcount(state)
    subsystem_size = wrt.bit_count()
    if state_size > subsystem_size:
        trace_tensor = state_size*[I]
        for i in range(state_size):
            if wrt&(1<<i):
                trace_tensor[i] = ket01.T
        state = apply_gates(state, trace_tensor)
    return state

# encodes state into [[15,1,3]] Reed-Muller code
# state is an n-qubit state represented as a sparse matrix
# p is an optional argument indicating qubit position (1 by default)
# returns 15-qubit state
def rm_encode(state, p=15):
    zeros = sp.sparse.csc_array(([1],([0],[0])),shape=(128,128),dtype=np.complex_)
    state = sp.sparse.kron(zeros,state,format='coo')
    offset = p-15
    state = apply_gate_on(state, H, 0b10001011<<offset)
    state = CNOT(state, 1<<offset, 0b101010101010100<<offset)
    state = CNOT(state, 1<<(1+offset), 0b110011001100100<<offset)
    state = CNOT(state, 1<<(3+offset), 0b111100001110000<<offset)
    state = CNOT(state, 1<<(7+offset), 0b111111100000000<<offset)
    return state

# pre-computed logical states for Reed-Muller code
rm_encoded_state_0 = rm_encode(ket0)
rm_encoded_state_1 = rm_encode(ket1)
    
# decodes state from [[15,1,3]] Reed-Muller code
# state is an 15-qubit state represented as a sparse matrix
# p is an optional argument indicating state position (15 by default)
# returns 1-qubit state
def rm_decode(state, p=1):
    if p==1:
        state = stab_4[1] @ state @ stab_4[0]
        state = stab_3[0][1] @ state @ stab_3[0][0] + stab_3[1][1] @ state @ stab_3[1][0]
        state = stab_2[0][1] @ state @ stab_2[0][0] + stab_2[1][1] @ state @ stab_2[1][0]
        state = stab_1[0][1] @ state @ stab_1[0][0] + stab_1[1][1] @ state @ stab_1[1][0]
        state = flip_5[0][1] @ state @ flip_5[0][0] + flip_5[1][1] @ state @ flip_5[1][0]
        state = decode_left @ state @ decode_right
    else:
        offset = p-1
        state = CNOT(state, 1<<(7+offset), 0b111111100000000<<offset)
        state = CNOT(state, 1<<(3+offset), 0b111100001110000<<offset)
        state = CNOT(state, 1<<(1+offset), 0b110011001100100<<offset)
        state = CNOT(state, 1<<offset, 0b101010101010100<<offset)
        state = CNOT(state, 1<<(14+offset), 0b101100110100<<offset)
        state = apply_gate_on(state, H, 0b10001011<<offset)
        state = partial_trace(state, 16383<<offset) #0b11111111111111
    return state

# performs checks for Reed-Muller code
# assumes encoded state is not entangled with rest of system
# state is n-qubit state on row p (1 by default)
# returns measurement results in two formats based on optional format argument
# 1) results stored REVERSE ORDER as bits in two integers for fast converting with error correction
# 2) two arrays, one for the measurement results of the Z stabilizers and one for the X stabilizers
def check(state, p=1, format='bits'):
    if qcount(state) != 15:
       state = partial_trace(state, ((1<<qcount(state))-1)<<(14+p)|((1<<(p-1)-1)))
    state = square_mat(state).tocsc()
    Zchecks = 0 if format=='bits' else np.zeros(10,dtype=np.complex_)
    Xchecks = 0 if format=='bits' else np.zeros(4,dtype=np.complex_)
    state_c = state.conj().tocsc()
    if format=='bits':
        for i in range(10):
            Zchecks |= ((state_c*(Zstabs[i][1] @ state @ Zstabs[i][0])).sum().real < 0)<<i
        for i in range(4):
            Xchecks |= ((state_c*(Xstabs[i][1] @ state @ Xstabs[i][0])).sum().real < 0)<<i
    else:
        for i in range(10):
            Zchecks[i] = (state_c*(Zstabs[i][1] @ state.tocsc() @ Zstabs[i][0])).sum()
        for i in range(4):
            Xchecks[i] = (state_c*(Xstabs[i][1] @ state.tocsc() @ Xstabs[i][0])).sum()
    return Zchecks, Xchecks

# file storing lookup table for Zchecks and Xchecks
# create lookup table if it doesn't not already exist
if not os.path.exists('Zcheck_lookup.npy'):
	# code used to create error lookup dictionary for Zchecks

	rm_state = rm_encoded_state_0
	# fill list with 1024 entries that are the least optimal error 
			                    #0b111111111111111
	error_mapping = [32767 for _ in range(1024)]

	# use bit array to indicate bit flip positions
	# leftmost is 15th qubit, rightmost is 1st qubit
	for i in range(0,32768):
		# apply X gates (bit flips) only on the bits that differ between iterations
		if i: rm_state = apply_gate_on(rm_state,X,(i-1)^i)
		# perform checks
		Zchecks, Xchecks = check(rm_state)
		# calculate 3 conditions for most likely error
		# 1) smallest number of qubits affected
		# 2) smallest distance between leftmost and rightmost affected qubits
		# 3) error closest to original encoded qubit (15th qubit)
		i_cond = (i.bit_count(), bit_distance(i), i.bit_length())
		curr_error = error_mapping[Zchecks]
		curr_cond = (curr_error.bit_count(), bit_distance(curr_error), curr_error.bit_length())
		# check conditions in order of precedence, store optimal result
		if ((i_cond[0] < curr_cond[0]) or
			(i_cond[0] == curr_cond[0] and i_cond[1] < curr_cond[1]) or
			(i_cond[0] == curr_cond[0] and i_cond[1] == curr_cond[1] and i_cond[2] > curr_cond[2])):
			error_mapping[Zchecks] = i
			
	error_mapping = np.array(error_mapping)
	np.save('Zcheck_lookup.npy', error_mapping)

# Z checks map based on previously mentioned criteria
Zcheck_decode = np.load('Zcheck_lookup.npy')
# X checks map directly because each set of measurements corresponds to a unique 1-qubit error
Xcheck_decode = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])

# code used for timing benchmark
def benchmark(n=10):
    avg = 0
    for i in range(n):
        start_time = time.perf_counter()
        
        rm_state = rm_encoded_state_0
        for i in range(0,15):
            rm_state = apply_gate_on(rm_state, X, 1<<i)
            Zchecks, Xchecks = check(rm_state)
            rm_state = apply_gate_on(rm_state, X, 1<<i)
        for i in range(0,15):
            rm_state = apply_gate_on(rm_state, Z, 1<<i)
            Zchecks, Xchecks = check(rm_state)
            rm_state = apply_gate_on(rm_state, Z, 1<<i)
        rm_state = rm_decode(rm_state)
        end_time = time.perf_counter()-start_time
        avg += end_time/n
    print(f'Avg:{avg}')
    
# code used for debugging Reed-Muller encoding/decoding, checks, logical operators
def debug_run():
    rm_state = rm_encoded_state_0
    # make sure checks are working for single-qubit errors
    for i in range(0,15):
        rm_state = apply_gate_on(rm_state, X, 1<<i)
        Zchecks, Xchecks = check(rm_state)
        print(bits_to_array(Zcheck_decode[Zchecks]), bits_to_array(Xcheck_decode[Xchecks]))
        rm_state = apply_gate_on(rm_state, X, 1<<i)
    for i in range(0,15):
        rm_state = apply_gate_on(rm_state, Z, 1<<i)
        Zchecks, Xchecks = check(rm_state)
        print(bits_to_array(Zcheck_decode[Zchecks]), bits_to_array(Xcheck_decode[Xchecks]))
        rm_state = apply_gate_on(rm_state, Z, 1<<i)
    rm_state = rm_decode(rm_state)
    # make sure state returns to |0>
    print(vec(rm_state))
    # make sure logical X works
    rm_state = rm_encode(rm_state)
    rm_state = apply_gates(rm_state, 15*[X])
    rm_state = rm_decode(rm_state)
    print(vec(rm_state))
    # make sure logical Z works
    rm_state = rm_encode(rm_state)
    rm_state = apply_gates(rm_state, 15*[Z])
    rm_state = rm_decode(rm_state)
    print(vec(rm_state))

# perform a single QEC cycle
def single_run(error_rate=0.1, noise_seed=None, model='cc', ph_error=0.1, ph_rep=3, ph_seed=None):
    rng = np.random.default_rng(seed=noise_seed)
    # generate arrays that represent where errors are applied
    # 0 means apply identity, 1 means apply error
    Xrand = rng.choice(2, 15, p=[1-error_rate, error_rate])
    Zrand = rng.choice(2, 15, p=[1-error_rate, error_rate])
    Xerrors = 0
    Zerrors = 0
    # convert array to bit array
    # leftmost is 15th qubit, rightmost is 1st qubit
    for i in range(15):
        Xerrors |= Xrand[i]<<i
        Zerrors |= Zrand[i]<<i
    # prepare state
    rm_state = rm_encoded_state_0
    # apply errors
    rm_state = apply_gate_on(rm_state, X, Xerrors)
    rm_state = apply_gate_on(rm_state, Z, Zerrors)
    Zchecks = 0
    Xchecks = 0
    # if error model is phenomenological, simulate measurement errors
    if model=='ph':
        reps = np.zeros(ph_rep,dtype=np.int_)
        meas_rng = np.random.default_rng(seed=ph_seed)
        # preform repeated measurements with error
        for i in range(ph_rep):
            combined_check = check(rm_state)
            combined_check = (combined_check[0]<<4)|(combined_check[1])
            rand_meas = rng.choice(2, 14, p=[1-ph_error, ph_error])
            for j in range(14):
                combined_check ^= (rand_meas[j])<<j
            reps[i] = combined_check
        majority_check = 0
        # for each stabilizer, pick the most probable result using majority voting
        for i in range(14):
            count = 0
            for rep in reps:
                count += 1&(rep>>i)
            majority_check |= (count > ph_rep//2)<<i
        Zchecks = majority_check >> 4
        Xchecks = majority_check & 15 #0b1111
    # otherwise, assume perfect measurements
    else:
        Zchecks, Xchecks = check(rm_state)
    # use lookup table to apply X and Z corrections
    rm_state = apply_gate_on(rm_state, X, Zcheck_decode[Zchecks])
    rm_state = apply_gate_on(rm_state, Z, Xcheck_decode[Xchecks])
    rm_state = rm_decode(rm_state)
    
    # count number of actual errors
    Xerror_ct = Xerrors.bit_count()
    Zerror_ct = Zerrors.bit_count()
    
    # X errors
    Xerror_type = 1 + rm_state.indices[0] - (Xerrors == Zcheck_decode[Zchecks])
                    # logical error		    # detected correct error
                    # if neither of the above, detected incorrect error but did not cause logical error

	# Z errors
    rm_state = apply_gate_on(apply_gate_on(rm_encoded_state_1, Z, Zerrors), Z, Xcheck_decode[Xchecks])
    Zerror_type = 4 + (rm_state.data[0].real < 0) - (Zerrors == Xcheck_decode[Xchecks])
    
    return Xerror_ct, Xerror_type, Zerror_ct, Zerror_type

# simulate a Pauli error channel
# n is number of runs
# error_rate is the chance of an error on one qubit
# noise_seed is the seed used for qubit errors
# model is the analysis model to use ('cc' = code capacity, 'ph'= phenomenological)
# ph_error sets the error rate of measurements
# ph_rep sets the number of measurements to do
# returns 2D array with results as specified below
def simulate_QEC(n=100, error_rate=0.1, noise_seed=None, model='cc', ph_error=0.1, ph_rep=3, ph_seed=None):
    # rows: # of actual qubit errors
	# columns:
	# - 0: Correctly detected X error 
	# - 1: Incorrectly detected X error that did not cause logical error
	# - 2: Logical X error
	# - 3: Correctly detected Z error 
	# - 4: Incorrectly detected Z error that did not cause logical error
	# - 5: Logical Z error
    errors = np.zeros((16,6),dtype=int)
    
    results = None
    with mp.Pool(processes=mp.cpu_count()-1) as p:
        results = p.starmap(single_run, n*[(error_rate, noise_seed, model, ph_error, ph_rep, ph_seed)])
    # store results of runs into errors array
    for r in results:
        errors[r[0]][r[1]] += 1
        errors[r[2]][r[3]] += 1
    return errors