from reed_muller import *
import matplotlib.pyplot as plt

# check if previous simulation results already exist
if not os.path.exists('sim_results.npy'):
    # if not, do runs
    start_time = time.perf_counter()
    runs = 1000
    steps = 10
    error_results = (steps-1)*[None]

    for i in range(1,steps):
        error_results[i-1] = simulate_QEC(n=runs, error_rate=np.round(i/steps,2))
                
    error_results = np.array(error_results)

    np.save('sim_results.npy', error_results)

    end_time = time.perf_counter()-start_time
    print(end_time)

# load simulation results        
error_results = np.load('sim_results.npy')
# save to Matlab file
sp.io.savemat("results_cc.mat", {'error_results':error_results})
# a Matlab file is used to make plots because the plots in Matlab are of higher quality