# Data Analytics Research Paper Code
This repository contains the code for the work done on the topic of bayesian networks for censorded data.
The work involve creating IPCW and Zupan weights which are then fed into the bayesian network as parameters 
for the structural and parameter learning algorithms. 

## *The description of the files contained in the repository:*  
**inverse.r** - contains the codes for the IPCW weights calculation  
**zupan.r** - contains the code for the Zupan Weights Calculation  
**bndata.csv** - original stanford heart transplant dataset  
**bndata_inverse.csv** - modified version of the data saved after the calculation of weights in the inverse.R file. This includes the IPCW weights and other extra columns in the dataset.  
**bndata_zupan.csv** - modified version of the data saved after the calculation of weights in the zupan.R file. This includes the zupan weights and other extra columns in the dataset.  
**ipwcBNs.py** - contains the code for the building of the Bayesian Network using the IPCW weights and the collection of the performance metrics.  
**zupanBNs.py** - contains the code for the building of the Bayesian Network using the Zupan weights and the collection of the performance metrics. 
**MICS** - inverseCorrected.r , inverseRestricted.r and inverseWithWeights.r are other libraries for bayesin network learning that are built in R. 


