import { kMeansClustering } from "./clusteringAlgos/kmeansClustering.js";
import { dbscan } from "./clusteringAlgos/dbScanClustering.js";
import { gmm } from "./clusteringAlgos/gmmClustering.js";
import { hierarchicalClustering } from "./clusteringAlgos/hClustering.js";
import { loadIrisDataset } from "./dataset.js";
// Example dataset (each row is a 2D point)
const data = await loadIrisDataset();

// Run K-Means with k=3
await kMeansClustering(data, 3).then(({ assignments, centroids }) => {
  console.log("Cluster Assignments:", assignments);
  console.log("Centroids:", centroids);
});

// const hierarchicalClusters = await hierarchicalClustering(data, 3);
// console.log("Final hierarchicalClusters: ", hierarchicalClusters);

const dbScanClusters = dbscan(data, 0.5, 5);
console.log("DBSCAN Cluster Assignments:", dbScanClusters);

const gmmClusters = await gmm(data, 3);
console.log("GMM Cluster Assignments:", gmmClusters);
