The notebook contains only the graphs. 

The 4 other python scripts are used to define the environment, the policies, the algorithms updates and the experiments. 

The experiments.py file have several functions to compute the projection matrices, the MSPBE, the followon trace,... The exp class defined there contains one major function that
do the episodes and get trace of the theta values. Since several types of errors interested us for the graphs, I just saved the thetas values and the mse computations will be done later. The other functions are written to make the experiments faster since I tried the same experiment but changing one parameter at a time. 
