from reed_muller import *

rm_state = sp.sparse.coo_array(([1],([0],[0])),shape=(2**8,2**7),dtype=np.complex_)
rm_state = rm_encode(rm_state)
random_positions = []
for i in range(1,16):
    if np.random.random() > 0.5:
        random_positions.append(i)
    
rm_state = applyGateOn(rm_state, X, random_positions)
rm_state = rm_decode(rm_state)
print(vec(rm_state))

#benchmark()