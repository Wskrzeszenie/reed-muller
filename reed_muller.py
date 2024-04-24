import functools as ft
import numpy as np
import scipy as sp
import time

# for the scipy library, COO is faster for general storage, reshaping and tensor products
#                        CSR/CSC is faster for arithmetic

# state kets for |0> and |1>
ket0 = sp.sparse.coo_array(([1],([0],[0])),shape=(2,1),dtype=np.complex_)
ket1 = sp.sparse.coo_array(([1],([1],[0])),shape=(2,1),dtype=np.complex_)

# density matrices for |0> and |1>
ZERO = sp.sparse.coo_array(np.array([[1,0],[0,0]],dtype=np.complex_)) 
ONE = sp.sparse.coo_array(np.array([[0,0],[0,1]],dtype=np.complex_))

# Pauli gates and H
I = sp.sparse.coo_array(np.array([[1,0],[0,1]],dtype=np.complex_))
X = sp.sparse.coo_array(np.array([[0,1],[1,0]],dtype=np.complex_))
Y = sp.sparse.coo_array(np.array([[0,-1j],[1j,0]],dtype=np.complex_))
Z = sp.sparse.coo_array(np.array([[1,0],[0,-1]],dtype=np.complex_))
H = sp.sparse.coo_array(np.array([[1,1],[1,-1]],dtype=np.complex_)/np.sqrt(2))

# lookup table for converting Zchecks to Z error positions
# multiple errors may map to the same measurements
# the most likely error is picked based on 3 factors in the following order of precedence
# 1) smallest number of qubits affected
# 2) smallest distance between leftmost and rightmost qubits (1st qubit is leftmost, 15th qubit is rightmost)
# 3) error closest to original encoded qubit (1st qubit)
Zcheck_decode = {0: 0, 1: 1, 2: 2, 3: 3, 19: 4, 18: 5, 17: 6, 16: 7, 4: 8, 5: 9, 6: 10, 7: 11, 23: 12, 22: 13, 21: 14, 20: 112, 37: 16, 36: 17, 39: 18, 38: 19, 54: 20, 55: 21, 52: 22, 53: 104, 33: 24, 32: 25, 35: 26, 34: 100, 50: 28, 51: 98, 48: 97, 49: 96, 70: 32, 71: 33, 68: 34, 69: 35, 85: 36, 84: 37, 87: 38, 86: 88, 66: 40, 67: 41, 64: 42, 65: 84, 81: 44, 80: 82, 83: 81, 82: 80, 99: 48, 98: 49, 97: 50, 96: 76, 112: 52, 113: 74, 114: 73, 115: 72, 103: 56, 102: 70, 101: 69, 100: 68, 116: 67, 117: 66, 118: 65, 119: 64, 8: 128, 9: 129, 10: 130, 11: 131, 27: 132, 26: 133, 25: 134, 24: 1792, 12: 136, 13: 137, 14: 138, 31: 140, 45: 144, 44: 145, 47: 146, 62: 148, 41: 152, 40: 6400, 56: 24832, 57: 224, 78: 160, 79: 161, 76: 162, 93: 164, 74: 168, 72: 10752, 88: 20992, 90: 208, 107: 176, 104: 19456, 120: 13312, 123: 200, 108: 196, 125: 194, 126: 193, 127: 192, 521: 256, 520: 257, 523: 258, 522: 259, 538: 260, 539: 261, 536: 262, 537: 1664, 525: 264, 524: 265, 527: 266, 542: 268, 556: 272, 557: 273, 558: 274, 575: 276, 552: 280, 553: 6272, 569: 24704, 568: 352, 591: 288, 590: 289, 589: 290, 604: 292, 587: 296, 585: 21504, 601: 11264, 603: 336, 618: 304, 617: 12800, 633: 18944, 634: 328, 621: 324, 636: 322, 639: 321, 638: 320, 513: 384, 512: 385, 515: 386, 514: 1540, 530: 388, 531: 1538, 528: 1537, 529: 1536, 517: 392, 516: 6160, 532: 24592, 533: 1544, 548: 400, 549: 6152, 565: 24584, 564: 1552, 544: 6145, 545: 6144, 546: 24580, 547: 6146, 563: 24578, 562: 6148, 561: 24576, 560: 24577, 583: 416, 582: 24640, 598: 6208, 599: 1568, 614: 1600, 615: 6176, 631: 24608, 630: 448, 138: 512, 139: 513, 136: 514, 137: 515, 153: 516, 152: 517, 155: 518, 154: 1408, 142: 520, 143: 521, 140: 522, 157: 524, 175: 528, 174: 529, 173: 530, 188: 532, 171: 536, 170: 25600, 186: 7168, 187: 608, 204: 544, 205: 545, 206: 546, 223: 548, 200: 552, 202: 10368, 218: 20608, 216: 592, 233: 560, 234: 12544, 250: 18688, 249: 584, 238: 580, 255: 578, 252: 577, 253: 576, 130: 640, 131: 641, 128: 642, 129: 1284, 145: 644, 144: 1282, 147: 1281, 146: 1280, 134: 648, 132: 10272, 148: 20512, 150: 1288, 167: 656, 165: 20544, 181: 10304, 183: 1296, 196: 672, 198: 10248, 214: 20488, 212: 1312, 192: 10242, 193: 20484, 194: 10240, 195: 10241, 211: 20481, 210: 20480, 209: 10244, 208: 20482, 229: 1344, 231: 10256, 247: 20496, 245: 704, 643: 768, 642: 769, 641: 770, 640: 1156, 656: 772, 657: 1154, 658: 1153, 659: 1152, 647: 776, 644: 18496, 660: 12352, 663: 1160, 678: 784, 677: 12320, 693: 18464, 694: 1168, 709: 800, 710: 12304, 726: 18448, 725: 1184, 736: 18436, 737: 12290, 738: 12289, 739: 12288, 755: 18432, 754: 18433, 753: 18434, 752: 12292, 740: 1216, 743: 12296, 759: 18440, 756: 832, 651: 896, 650: 1030, 649: 1029, 648: 1028, 664: 1027, 665: 1026, 666: 1025, 667: 1024, 652: 1036, 669: 1034, 670: 1033, 671: 1032, 685: 1044, 700: 1042, 703: 1041, 702: 1040, 682: 1120, 683: 6656, 699: 25088, 698: 1048, 718: 1060, 735: 1058, 732: 1057, 733: 1056, 713: 1104, 715: 10496, 731: 20736, 729: 1064, 744: 1096, 747: 12416, 763: 18560, 760: 1072, 748: 1088, 749: 1089, 750: 1090, 767: 1092, 268: 2048, 269: 2049, 270: 2050, 271: 2051, 287: 2052, 286: 2053, 285: 2054, 284: 28672, 264: 2056, 265: 2057, 266: 2058, 283: 2060, 297: 2064, 296: 2065, 299: 2066, 314: 2068, 301: 2072, 300: 4480, 316: 5632, 317: 2144, 330: 2080, 331: 2081, 328: 2082, 345: 2084, 334: 2088, 332: 8832, 348: 9472, 350: 2128, 367: 2096, 364: 17536, 380: 17152, 383: 2120, 360: 2116, 377: 2114, 378: 2113, 379: 2112, 260: 2176, 261: 2177, 262: 2178, 279: 2180, 256: 2184, 257: 4368, 258: 8736, 275: 17472, 289: 2192, 288: 4360, 290: 17440, 307: 8768, 293: 4353, 292: 4352, 294: 4354, 311: 4356, 322: 2208, 320: 8712, 321: 17424, 339: 4416, 326: 8706, 324: 8704, 325: 8705, 343: 8708, 358: 17410, 357: 17409, 356: 17408, 375: 17412, 354: 4384, 353: 8720, 352: 17416, 371: 2240, 773: 2304, 772: 2305, 775: 2306, 790: 2308, 769: 2312, 768: 4240, 770: 16960, 787: 9248, 800: 2320, 801: 4232, 802: 9280, 819: 16928, 804: 4225, 805: 4224, 807: 4226, 822: 4228, 835: 2336, 848: 16912, 849: 9224, 850: 4288, 838: 9220, 852: 9217, 853: 9216, 855: 9218, 870: 16900, 885: 16896, 884: 16897, 887: 16898, 867: 4256, 881: 16904, 880: 9232, 882: 2368, 781: 2432, 780: 4120, 796: 4192, 797: 3584, 777: 4113, 776: 4112, 778: 4114, 795: 4116, 808: 4105, 809: 4104, 811: 4106, 826: 4108, 812: 4097, 813: 4096, 814: 4099, 815: 4098, 831: 4101, 830: 4100, 829: 26624, 828: 4102, 841: 4164, 856: 4162, 858: 4160, 859: 4161, 846: 4144, 845: 8960, 861: 9344, 862: 4168, 879: 4136, 877: 17664, 893: 17024, 895: 4176, 874: 4129, 875: 4128, 873: 4130, 888: 4132, 390: 2560, 391: 2561, 388: 2562, 405: 2564, 386: 2568, 384: 8352, 385: 16704, 403: 5136, 419: 2576, 432: 16672, 433: 8384, 434: 5128, 421: 5124, 436: 5122, 438: 5120, 439: 5121, 448: 2592, 449: 5184, 450: 8328, 467: 16656, 452: 8322, 454: 8320, 455: 8321, 469: 8324, 485: 16644, 502: 16640, 503: 16641, 500: 16642, 483: 8336, 498: 16648, 496: 5152, 497: 2624, 398: 2688, 396: 8232, 412: 8272, 414: 3328, 394: 8226, 392: 8224, 393: 8225, 411: 8228, 426: 8260, 440: 8257, 441: 8256, 443: 8258, 430: 4864, 429: 8240, 445: 8264, 446: 5248, 456: 8202, 458: 8200, 459: 8201, 473: 8204, 460: 8194, 461: 8195, 462: 8192, 463: 8193, 479: 8198, 478: 22528, 477: 8196, 476: 8197, 495: 8216, 494: 17920, 510: 16768, 511: 8288, 489: 8210, 491: 8208, 490: 8209, 504: 8212, 911: 2816, 908: 16456, 924: 16432, 927: 3200, 906: 16450, 905: 16449, 904: 16448, 923: 16452, 938: 16420, 953: 16416, 952: 16417, 955: 16418, 943: 4736, 941: 16464, 957: 16424, 959: 5376, 969: 16404, 986: 16400, 987: 16401, 984: 16402, 975: 8576, 974: 16480, 990: 16408, 991: 9728, 1004: 16388, 1005: 16389, 1006: 16390, 1007: 14336, 1023: 16384, 1022: 16385, 1021: 16386, 1020: 16387, 1000: 16396, 1019: 16392, 1018: 16393, 1017: 16394, 900: 3076, 917: 3074, 918: 3073, 919: 3072, 898: 4624, 897: 8480, 896: 16576, 915: 3080, 931: 4616, 945: 16544, 944: 8512, 946: 3088, 934: 4609, 935: 4608, 933: 4610, 948: 4612, 963: 8456, 978: 16528, 976: 4672, 977: 3104, 965: 8450, 967: 8448, 966: 8449, 980: 8452, 996: 16516, 1015: 16512, 1014: 16513, 1013: 16514, 992: 3136, 993: 4640, 994: 8464, 1011: 16520}

# perform tensor product on a list of matrices
# returns resulting tensor product
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
    r,c = matrix.shape
    return matrix.reshape((r*c,1),order='F')

# converts a vector/matrix of any shape to a square matrix
# returns square matrix
# if matrix is not square-able, does approximate square matrix with dimensions (2n,n)
def square_mat(matrix):
    entries = qcount(matrix)
    return matrix.reshape((1<<(entries-entries//2),1<<(entries//2)),order='F')

# returns purity of state
#def purity(state):
    
# counts number of qubits represented by sparse matrix
# state is an n-qubit state represented as a sparse matrix
# returns number of qubits in state based on size
def qcount(state):
    return int(np.log2(np.prod(state.shape)))

# applies a list of 15 gates to the state
# state is an n-qubit state represented as a sparse matrix
# returns state after applying gates
def apply_gates(state, gates):
    return (tensor(gates[len(gates)//2:]).tocsc() @ state.tocsc() @ tensor(gates[:len(gates)//2]).tocsc().conj().T)

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
# assumes state is NOT mixed
def partial_trace(state, wrt):
    system_size = qcount(state)
    subsystem_size = len(wrt)
    if state.size() == 1:
        bits = state.shape
        state.row[0]

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
    
# decodes state from [[15,1,3]] Reed-Muller code
# state is an 15-qubit state represented as a sparse matrix
# p is an optional argument indicating state position (15 by default)
# returns 1-qubit state
def rm_decode(state, p=1):
    offset = p-1
    state = CNOT(state, 1<<(7+offset), 0b111111100000000<<offset)
    state = CNOT(state, 1<<(3+offset), 0b111100001110000<<offset)
    state = CNOT(state, 1<<(1+offset), 0b110011001100100<<offset)
    state = CNOT(state, 1<<offset, 0b101010101010100<<offset)
    state = CNOT(state, 1<<(14+offset), 0b101100110100<<offset)
    state = apply_gate_on(state, H, 0b10001011<<offset)
    return state

# performs checks for Reed-Muller code
# assumes encoded state is not entangled with rest of system
# input is n-qubit state on row p (1 by default)
# output is measurement results
# returns in two formats based on optional format argument
# 1) results stored REVERSE ORDER as bits in two integers for fast converting with error correction
# 2) two arrays, one for the measurement results of the Z stabilizers and one for the X stabilizers
def check(state, p=1, format='bits'):
    if qcount(state) != 15:
        state = partial_trace(state)
    state = square_mat(state).tocsc()
    Zchecks = 0 if format=='bits' else np.zeros(10,dtype=np.complex_)
    Xchecks = 0 if format=='bits' else np.zeros(4,dtype=np.complex_)
    state_c = state.conj()
    if format=='bits':
        for i in range(10):
            if ((state_c*(Zstabs[i][1] @ state.tocsc() @ Zstabs[i][0])).sum().real < 0):
                Zchecks |= 1<<i
        for i in range(4):
            if ((state_c*(Xstabs[i][1] @ state.tocsc() @ Xstabs[i][0])).sum().real < 0):
                Xchecks |= 1<<i
    else:
        for i in range(10):
            Zchecks[i] = (state_c*(Zstabs[i][1] @ state.tocsc() @ Zstabs[i][0])).sum()
        for i in range(4):
            Xchecks[i] = (state_c*(Xstabs[i][1] @ state.tocsc() @ Xstabs[i][0])).sum()
    return Zchecks, Xchecks

# code used for timing benchmark
# optional inputs are number of runs, print/not print checks, print/not print time for each runs
def benchmark(n=10, print_checks=False, print_times=False):
    avg = 0
    for i in range(n):
        start_time = time.perf_counter()
        
        rm_state = ket0.copy()
        rm_state = rm_encode(rm_state)
        for i in range(0,15):
            rm_state = apply_gate_on(rm_state,X,1<<i)
            Zchecks,Xchecks = check(rm_state)
            if print_checks:
                print(bits_to_array(Zcheck_decode[Zchecks]),Xchecks)
            rm_state = apply_gate_on(rm_state,X,1<<i)
        for i in range(0,15):
            rm_state = apply_gate_on(rm_state,Z,1<<i)
            Zchecks,Xchecks = check(rm_state)
            if print_checks:
                print(bits_to_array(Zcheck_decode[Zchecks]),Xchecks)
            rm_state = apply_gate_on(rm_state,Z,1<<i)
        rm_state = rm_decode(rm_state)
        end_time = time.perf_counter()-start_time
        if print_times:
            print(end_time)
        avg += end_time/n
    print(f"Avg:{avg}")