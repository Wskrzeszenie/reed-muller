from reed_muller import *

benchmark(n=20)
#debug_run()
'''
runs = 10
steps = 20
error_results = (steps-1)*[None]

for i in range(1,steps):
	error_results[i-1] = simulate_QEC(n=runs, error_rate=np.round(i/20,2))
			
error_results = np.array(error_results)

np.save('results.npy', error_results)

error_results = np.load('results.npy')
print(error_results)
'''
