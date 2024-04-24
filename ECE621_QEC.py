from reed_muller import *
from Zcheck_decoder import *

'''
count = 0

for key in Zcheck_decode_lists.keys():
    possible_errors = Zcheck_decode_lists[key]
    for i in range(len(possible_errors)):
        possible_errors[i] = (bit_distance(possible_errors[i]), 15-possible_errors[i].bit_length(), possible_errors[i])
    possible_errors.sort()

    if len(possible_errors) != 1 and possible_errors[0][0] == possible_errors[1][0] and possible_errors[0][1] == possible_errors[1][1]:
        print("Ambiguous check: ", key, possible_errors)
        count += 1

print(count)
'''
benchmark(n=20)