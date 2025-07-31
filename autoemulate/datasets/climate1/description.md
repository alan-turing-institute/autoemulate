# climate1

* data: from Richard Wilkinson
* details: https://www.nature.com/articles/s41558-018-0197-7 (probably GENIE model)
* contact: Richard Wilkinson (University of Nottingham)

### Description:
The aim is to predict the climate variables SAT, ACC, VEGC, SOILC, MAXPMOC, OCN_O2, fCaCO3, SIAREA_S from the  variables in columns 2 to 34 (indexing from 1). 

Note: We found in the paper that considering different mean structures and covariance functions for each different output produced good results. This included using the LASSO to select a subset of input variables.