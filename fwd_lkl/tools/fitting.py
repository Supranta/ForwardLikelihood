import numpy as np

def write_catalog_parameters(f, i, n, opt_value, uncertainty_arr, catalog_objs):
        f.write('\n Catalog %d: \n'%(i))
        catalog_obj = catalog_objs[i]
        for j in range(catalog_obj.num_params()):
                f.write('parameter%d : %2.4f +/- %2.4f\n'%(j,opt_value[n+j],uncertainty_arr[n+j]))
        n += catalog_obj.num_params()
        return n

def sample(sampler, pos0, N_MCMC, output_dir):
        i = 0
        start_time = time.time()
        for result in sampler.sample(pos0, iterations=N_MCMC):
                end_time = time.time()
                print("Current Iteration: "+str(i)+", Time Taken: %2.2f \n"%(end_time - start_time))
                i += 1
                start_time = time.time()
                if(i%10==0):
                    np.save(output_dir+'/chain.npy',sampler.chain[:,:i,:])
        np.save(output_dir+'/chain.npy',sampler.chain)

def uncertainty(f, opt, args, eps):
    print('Finding uncertainty of the parameters....')
    uncertainty_arr = np.zeros(len(opt))

    for i in range(len(opt)):
        x1 = np.array(opt)
        x1[i] = x1[i] - 2*eps[i]
        x2 = np.array(opt)
        x2[i] = x2[i] - eps[i]
        x3 = np.array(opt)
        x4 = np.array(opt)
        x4[i] = x4[i] + eps[i]
        x5 = np.array(opt)
        x5[i] = x5[i] + 2*eps[i]

        y1 = f(x1, *args)
        y2 = f(x2, *args)
        y3 = f(x3, *args)
        y4 = f(x4, *args)
        y5 = f(x5, *args)

        a, b, c = np.polyfit(np.array([x1[i], x2[i], x3[i], x4[i], x5[i]]), np.array([y1, y2, y3, y4, y5]), 2)

        uncertainty_arr[i] = 1/np.sqrt(a*2.0)

    return uncertainty_arr
