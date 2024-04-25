from reed_muller import *

# code used to create error lookup dictionary for Zchecks

rm_state = ket0.copy()
rm_state = rm_encode(rm_state)

# fill list with 1024 entries that are the least optimal error 
                            #0b111111111111111
error_mapping = [32767 for _ in range(1024)]

# use bits of number to indicate bit flip positions
# leftmost is 1st qubit, rightmost is 15th qubit
for i in range(0,32768):
    # apply X gates (bit flips) only on the bits that differ between iterations
    if i: rm_state = apply_gate_on(rm_state,X,(i-1)^i)
    # perform checks
    Zchecks,Xchecks = check(rm_state)
    # calculate 3 conditions for most likely error
    # 1) smallest number of qubits affected
    # 2) smallest distance between leftmost and rightmost qubits (1st qubit is leftmost, 15th qubit is rightmost)
    # 3) error closest to original encoded qubit (1st qubit)
    i_cond = (i.bit_count(), bit_distance(i), i.bit_length())
    curr_cond = (error_mapping[Zchecks].bit_count(), bit_distance(error_mapping[Zchecks]), error_mapping[Zchecks].bit_length())
    # check conditions in order of precedence, store optimal result
    if ((i_cond[0] < curr_cond[0]) or
        (i_cond[0] == curr_cond[0] and i_cond[1] < curr_cond[1]) or
        (i_cond[0] == curr_cond[0] and i_cond[1] == curr_cond[1] and i_cond[2] < curr_cond[2])):
        error_mapping[Zchecks] = i
        
print(error_mapping)
print(len(error_mapping))
