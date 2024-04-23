import numpy as np

Zstabs=np.array([0b101010101010101  #ZIZIZIZIZIZIZIZ
				,0b011001100110011  #IZZIIZZIIZZIIZZ
				,0b000111100001111  #IIIZZZZIIIIZZZZ
				,0b000000011111111  #IIIIIIIZZZZZZZZ
				,0b001000100010001  #IIZIIIZIIIZIIIZ
				,0b000010100000101  #IIIIZIZIIIIIZIZ
				,0b000001100000011  #IIIIIZZIIIIIIZZ
				,0b000000000110011  #IIIIIIIIIZZIIZZ
				,0b000000000001111  #IIIIIIIIIIIZZZZ
				,0b000000001010101  #IIIIIIIIZIZIZIZ
				])

Xstabs=np.array([0b101010101010101  #XIXIXIXIXIXIXIX
				,0b011001100110011  #IXXIIXXIIXXIIXX
				,0b000111100001111  #IIIXXXXIIIIXXXX
				,0b000000011111111  #IIIIIIIXXXXXXXX
				])

error_sets = []
error_bits = []
error_bits_set = set()

for i in range(1,16):
	stabs = []
	stabs_bitrep = 0
	for j in range(10):
		if (1<<(15-i))&Zstabs[j]:
			stabs.append(j+1)
			stabs_bitrep += 1<<j
	print(stabs)
	print(bin(stabs_bitrep))
	error_sets.append(stabs)
	error_bits.append(stabs_bitrep)
	error_bits_set.add(stabs_bitrep)
	
for i in range(10):
	for j in range(i,10):
		if (error_bits[i]&error_bits[j])==0 and error_bits[i]|error_bits[j] not in error_bits_set:
			print(error_sets[i],error_sets[j])
