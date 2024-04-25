from reed_muller import *

# file for benchmarking CSR vs CSC format
# ultimately, CSC is better suited due to column vector nature of quantum computing

csr = 0;
csc = 0;

for i in range(100):
    start_time = time.perf_counter()
    initial_csr = sp.sparse.csr_array(([1],([0],[0])),shape=(128,256),dtype=np.complex_)
    final_csr = tensor([I,I,I,I,I,I,I]).tocsr() @ initial_csr @ tensor([X,X,X,X,X,X,X,X]).tocsr()
    final_csr = final_csr.reshape((32768,1),order='C')
    end_time_csr = time.perf_counter()-start_time
 
    start_time = time.perf_counter()
    initial_csc = sp.sparse.csc_array(([1],([0],[0])),shape=(256,128),dtype=np.complex_)
    final_csc = tensor([X,X,X,X,X,X,X,X]).tocsc() @ initial_csc @ tensor([I,I,I,I,I,I,I]).tocsc()
    final_csc = final_csc.reshape((32768,1),order='F')
    end_time_csc = time.perf_counter()-start_time
    if end_time_csr > end_time_csc:
        csr += 1
    else:
        csc += 1
        
print(csr,csc)
