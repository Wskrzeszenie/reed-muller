from reed_muller import *

rm_state = ket0.copy()
rm_state = rm_encode(rm_state)
error_mapping = {}
for i in range(0,32768):
	positions = []
	for j in range(15):
		if 1&(i>>j):
			positions.append(j+1)
	if positions:
		rm_state = applyGateOn(rm_state,X,positions)
		Zchecks,Xchecks = check(rm_state)
		rm_state = applyGateOn(rm_state,X,positions)
	check_bits = 0
	for j in range(len(Zchecks)):
		if Zchecks[j].real < 0:
			check_bits += 1<<j
	if check_bits in error_mapping:
		if i.bit_count() == error_mapping[check_bits].bit_count():
			error_mapping[check_bits] = -1
		elif i.bit_count() < error_mapping[check_bits].bit_count():
			error_mapping[check_bits] = i
	else:
		error_mapping[check_bits] = i
	if i%1024==0: print(i)

for key in list(error_mapping.keys()):
	if error_mapping[key] = -1:
		del error_mapping[key]
