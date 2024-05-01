from reed_muller import *

if __name__ == '__main__':
    # check if previous simulation results already exist
    if not os.path.exists('sim_results_cc.npy'):
        # if not, do runs
        start_time = time.perf_counter()
        runs = 100
        steps = 20
        error_results = (steps-1)*[None]

        for i in range(1,steps):
            error_results[i-1] = simulate_QEC(n=runs, error_rate=np.round(i/steps,2))
                    
        error_results = np.array(error_results)
        np.save('sim_results_cc.npy', error_results)

        end_time = time.perf_counter()-start_time
        print(end_time)
    if not os.path.exists('sim_results_ph_qubit.npy'):
        start_time = time.perf_counter()
        runs = 100
        steps = 20
        error_results = (steps-1)*[None]

        for i in range(1,steps):
            error_results[i-1] = simulate_QEC(n=runs, error_rate=np.round(i/steps,2), model='ph', ph_error=0.1)
                    
        error_results = np.array(error_results)
        np.save('sim_results_ph_qubit.npy', error_results)

        end_time = time.perf_counter()-start_time
        print(end_time)
    if not os.path.exists('sim_results_ph_meas.npy'):
        start_time = time.perf_counter()
        runs = 100
        steps = 20
        error_results = (steps-1)*[None]

        for i in range(1,steps):
            error_results[i-1] = simulate_QEC(n=runs, error_rate=0.1, model='ph', ph_error=np.round(i/steps,2))
                    
        error_results = np.array(error_results)
        np.save('sim_results_ph_meas.npy', error_results)

        end_time = time.perf_counter()-start_time
        print(end_time)
    if not os.path.exists('sim_results_ph_rep.npy'):
        start_time = time.perf_counter()
        runs = 100
        steps = 16
        error_results = (steps-1)*[None]

        for i in range(1,steps):
            error_results[i-1] = simulate_QEC(n=runs, error_rate=0.1, model='ph', ph_error=0.1, ph_rep=2*i-1)
                    
        error_results = np.array(error_results)
        np.save('sim_results_ph_rep.npy', error_results)

        end_time = time.perf_counter()-start_time
        print(end_time)
    
    # load simulation results        
    error_results_cc = np.load('sim_results_cc.npy')
    error_results_ph_qubit = np.load('sim_results_ph_qubit.npy')
    error_results_ph_meas = np.load('sim_results_ph_meas.npy')
    error_results_ph_rep = np.load('sim_results_ph_rep.npy')

    # save to Matlab file
    sp.io.savemat("results.mat", {'error_results_cc':error_results_cc, 'error_results_ph_qubit':error_results_ph_qubit, 'error_results_ph_meas':error_results_ph_meas, 'error_results_ph_rep':error_results_ph_rep})
    print(error_results_cc.sum(axis=1))
    print(error_results_qubit.sum(axis=1))
    print(error_results_ph_meas.sum(axis=1))
    print(error_results_ph_rep.sum(axis=1))
    # a Matlab file is used to make plots because the plots in Matlab are of higher quality