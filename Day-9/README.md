# Day-9: Clustering with K-Means and PCA

## What I worked on

Today's focus was unsupervised learning — grouping data without target labels, using K-Means clustering, and reducing high-dimensional data with PCA, using the Iris dataset built into Scikit-learn.

## Dataset exploration (dataset_exploration.py)

- Loaded the Iris dataset and converted it into a Pandas DataFrame.
- Explored it using head, info, describe, and checked the species distribution (50 samples each of setosa, versicolor, and virginica).
- Kept the species labels in the DataFrame only for later comparison against the clusters K-Means finds - K-Means itself never sees these labels during clustering, since clustering is unsupervised.

## What is clustering?

Clustering is an unsupervised learning technique that groups similar data points together based only on their features, without using any target labels. K-Means specifically works by picking K cluster centers, assigning every point to its nearest center, then repeatedly updating the centers based on the average position of the points assigned to them, until the assignments stop changing.

## K-Means clustering (kmeans_clustering.py)

- Scaled the features using StandardScaler before clustering, since K-Means relies on distance calculations, and unscaled features (where some have a wider range than others) would distort those distances.
- Used the Elbow Method: ran K-Means for K values from 1 to 10, and plotted the inertia (sum of squared distances from each point to its assigned cluster center) for each. Inertia always decreases as K increases, so the goal is to find the point where adding more clusters stops giving a meaningful improvement - the "elbow" in the curve.

### How I determined the best value of K

The elbow chart (elbow_method.png) shows a sharp drop in inertia from K=1 to K=3, after which the curve flattens out noticeably. This pointed to K=3 as the optimal number of clusters, which also matches the fact that the Iris dataset genuinely contains 3 species.

- Applied K-Means with K=3 and visualized the resulting clusters using petal length and petal width (the two features that separate the species most clearly).

## PCA (pca_analysis.py)

## What is PCA?

PCA (Principal Component Analysis) is a dimensionality reduction technique that transforms a dataset with many features into a smaller number of new features (called principal components), while preserving as much of the original variance (information) as possible. It's useful because high-dimensional data is hard to visualize directly, and some features may be redundant or correlated with each other.

- Scaled the features first, since PCA looks for directions of maximum variance, and would otherwise be skewed by features with a larger raw scale.
- Reduced the original 4 features down to 2 principal components.
- The 2 components retained about 95.81% of the original variance - meaning very little information was lost by dropping from 4 dimensions to 2.

## Mini Project: Iris Flower Clustering & Visualization (iris_clustering_project.py)

Combines the full workflow: dataset loading and exploration, K-Means clustering with the elbow method, PCA reduction to 2 dimensions, and a side-by-side visualization comparing:
1. The original data (colored by true species)
2. The K-Means cluster assignments
3. The PCA-transformed data (colored by true species)

All three are shown together in mini_project_comparison.png.

## What insights did I gain from the visualizations?

- **How many clusters were formed?** 3, matching the number of actual Iris species.
- **Did the clusters represent the flower species well?** Very well for setosa - it was separated perfectly, with all 50 samples landing in a single cluster and no other species mixed in, since its petal measurements are clearly distinct from the other two. Versicolor and virginica showed some genuine overlap: comparing the clusters against the true species (see the crosstab printed by the script), 39 versicolor samples and 14 virginica samples landed in one cluster, while 11 versicolor and 36 virginica landed in another. This makes sense, since these two species are more similar to each other in petal size than either is to setosa.
- **How did PCA help in visualization?** Reducing from 4 features down to 2 principal components made it possible to see the entire dataset's structure on a single 2D scatter plot, while still keeping about 96% of the original information. The PCA plot shows the same overall pattern as the original petal-based scatter plot - setosa clearly separated, versicolor and virginica closer together - confirming that PCA preserved the meaningful structure in the data rather than distorting it.

## Files

- dataset_exploration.py: loading and exploring the Iris dataset
- kmeans_clustering.py: K-Means clustering with the elbow method
- elbow_method.png, kmeans_clusters.png: charts from the clustering script
- pca_analysis.py: PCA dimensionality reduction
- pca_visualization.png: chart from the PCA script
- iris_clustering_project.py: full mini project combining clustering, PCA, and comparison
- mini_project_elbow_method.png, mini_project_comparison.png: charts from the mini project
- README.md: this file

## Author

Hifsa Iftikhar
