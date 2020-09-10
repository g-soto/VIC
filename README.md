# VIC.

The Validity Index using supervised Classifiers (VIC) [https://www.sciencedirect.com/science/article/abs/pii/S0950705118300091?via%3Dihub] gives a quality index for a partition 
given by a clustering algorithm. From a group of supervised classifiers, the biggest AUC on cross-validation for all the classifiers is used as a quality index of the partition. 
It uses as class label the cluster label of a given partition.

# Remove/include supervised classifiers.

To remove/include supervised classifiers you need to modify the get_model function at line 168 in the VIC.cs file. This function contains a switch statement, each case corresponds to a classifier. You can remove/include new cases to this switch. Be sure to set the appropriate number_of_models argument in the calls to the VIC algorithm at lines 233 and 235 of the VIC.cs file.

# Multi-threaded.

The parameter cores in the vic function (line 131, VIC.cs) allows to specify the desired number of thread used to run the algorithm.
