# Coffee-Quality-Analysis

A complete clustering analysis of a coffee beans characteristics dataset.<br>
The purpose of this project is to find out if coffee beans from different parts of the world have similar values across multiple variables and if we can actually group them in clusters based on the data of each type of bean.<br>

Before executing the clustering analysis a PCA (Principal Component Analysis) is done to determine the variables which are most informative (the best ones to use for the clustering process, since they determine most of the variability in the data).<br>


<h4>Main Libraries Used for The Project<br></h4>
<li>Scikit-Learn</li>
<li>MLXTend</li>
<li>SciPy</li>
<li>Yellowbrick</li>

<br>

<h4>Clustering Evaluation Methods</h4>
<li>Elbow Method</li>
<li>Silhouette</li>

<br>

<h4>Project Operation:</h4>

1. It retrieves the dataset
2. Cleans the data
3. Exports the cleaned data into a secondary CSV file
4. Executes an EDA to generate an overview of the data and prints the main descriptive statistics which are mandatory to know before starting the analysis
5. Preprocesses the data to prepare it for the analysis
6. Executes a PCA with two different solvers <br>
6.1 SVD (Singular Value Decomposition)<br>
6.2 Auto-Solver (Set by default by the libraries)
7. Exports the Scree Plots which represent the explained variance for each dimension
8. Executes clustering on the data using the K-Means algorithm
9. Analyzes the data obtained from the clustering process
10. Calculates the optimal number of clusters using the Elbow Method and plots the results
11. Calculates the optimal number of clusters using the Silhouette Method (which is more accurate)<br>
11.1 Three different distances are used: Euclidean, Minkowski and Manhattan
11.2 The function which implements the Silhouette Method also returns the best distance for each K number of clusters
12. The clustering results get represented through a 3D plot with three variables. One plot gets generated for each K<br>
<i>Example: if the analysis implies executing the clustering process with K from 2 to 10, then 9 plots will be generated, one for each K.</i>
13. The plots get exported and the clustering results printed on the terminal


<br><br><br>
Original source of the dataset: https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi


