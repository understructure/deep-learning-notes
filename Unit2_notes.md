
Principal Components Analysis - Five Steps


1. Normalize data
2. Compute covariance matrix - get covariance b/t each feature w/ every other feature
  - subtract mean, calculate transpose, multiply by feature matrix minus mean, then take whole feature and divide by number of features minus one
3. Eigen decomposition - get the eigenvectors and Eigen values
  - Eigenvectors are the principle components of a data set - give us directions among which our transformation acts
  - Eigen values give us the magnitude of each
  - Sort both in descending order, then create matrix out of them - use this matrix to transform our original feature matrix via the dot product
  - Can then plot these in 2D space and use these PCs to replace our many features



1. 3 steps to preprocesing a dataset - normalization, transformation and reduction
2. Deep learning - Architecture engineering is the new feature engineering
3. PCA is a popular dimensionality reduction technique that can be implemented with sklearn
