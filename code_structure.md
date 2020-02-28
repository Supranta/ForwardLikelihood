# Structure of the code

### Basic Usage

You can change the config file to determine, e.g: 
- which reconstruction fields to use
- which peculiar velocity data to compare
- Other stuff such as whether to use a lognormal probability for the distance, etc. 

Once the config file is set, you can use the following command to infer the best fit parameters,

```python
python3 fit.py CONFIG_FILE_PATH
```

If sampling from the distribution, this will save the mcmc chains in the output folder. You can then use the same config file to process the mcmc chains using the following command:

```python
python3 analyze.py CONFIG_FILE_PATH
```

### What are the parameters that are inferred and what are the options that are available:

You can split the inferred parameters into two categories:

- Flow parameters: The basic flow parameters are beta and V_ext. These parameters are always inferred. If you set `vary_sig_v = True` in the config file, sigma_v is inferred in addition to the other flow parameters.
- Peculiar velocity survey parameters: There are also parameters associated with each PV catalog that we may need to infer. Following are the possibilities with different data types:

    1. `simple_distance`: You can choose to add an additional intrinsic scatter if only measurement errors are reported. You could also rescale the distances for each object in the catalog by some parameter. 
    2. `tf`: This option uses the magnitude and the velocity width directly and infers the TF parameters, a_TF, b_TF and sigma_int
    3. `sn_lc_fit`: This option uses the Tripp parameter and infers the parameters, `M, alpha, beta_sn, sigma_int` as well.
    
The PV survey parameters are appended to the flow parameters and jointly inferred in the inference process. 

Another option that can be added is to use lognormal distance prior, i.e, use a Gaussian prior in log-distance instead of a Gaussian in the distance. This is the correct way of doing it, although in our experience, it doesn't change the parameter inference by much.
