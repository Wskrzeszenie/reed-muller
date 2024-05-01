from reed_muller import *

if __name__ == '__main__':
    # check if previous simulation results already exist
    runs = 10000
    if not os.path.exists('sim_results_cc.npy'):
        # if not, do runs
        start_time = time.perf_counter()
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
    
    # print table statistics
    print(" X_C  X_I  X_L  Z_C  Z_I  Z_L")
    print(error_results_cc.sum(axis=1)[1])
    print(error_results_ph_qubit.sum(axis=1)[1])
    print(error_results_ph_meas.sum(axis=1)[3])
    print(error_results_ph_rep.sum(axis=1)[2])
    
    # save to Matlab file
    sp.io.savemat("results.mat", {'error_results_cc':error_results_cc, 'error_results_ph_qubit':error_results_ph_qubit, 'error_results_ph_meas':error_results_ph_meas, 'error_results_ph_rep':error_results_ph_rep})
    print
    # a Matlab file is used to make plots because the plots in Matlab are of higher quality
    # The runtimes on my laptop for all four simulations were:
    # 316.9013156997971 s
    # 429.4783999999054 s
    # 431.7662402000278 s
    # 854.6052580999676 s