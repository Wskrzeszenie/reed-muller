from reed_muller import *

start_time = time.perf_counter()
initial_csr = sp.sparse.csr_array(([1],([0],[0])),shape=(32768,1),dtype=np.complex_)
final_csr = tensor([I,I,I,I,I,I,H,H,H,I,I,I,I,I,I]).tocsr() @ initial_csr
end_time = time.perf_counter()-start_time
print(final_csr)
print(end_time)

start_time = time.perf_counter()
initial_csc = sp.sparse.csc_array(([1],([0],[0])),shape=(32768,1),dtype=np.complex_)
final_csc = tensor([I,I,I,I,I,I,H,H,H,I,I,I,I,I,I]).tocsc() @ initial_csc
end_time = time.perf_counter()-start_time
print(final_csc)
print(end_time)


start_time = time.perf_counter()
initial = sp.sparse.csr_array(([1],([0],[0])),shape=(256,128),dtype=np.complex_)
final = applyGateOn(initial, H, [7,8,9])
end_time = time.perf_counter()-start_time
print(final)
print(end_time)
