import multiprocessing as mp
import time

def square(x):
    return x*x

if __name__ == '__main__':
    start_time = time.perf_counter()
    with mp.Pool() as pool:
        results = pool.map(square, range(16))
        print(results)
    end_time = time.perf_counter()-start_time
    print(end_time)
    start_time = time.perf_counter()
    s = list(range(16))
    for i in range(16):
        s[i] = square(i)
    print(s)
    end_time = time.perf_counter()-start_time
    print(end_time)