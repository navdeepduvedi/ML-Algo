import * as kMeansClustering from "./clusteringAlgos/kmeansClustering.js";
import * as dbscan from "./clusteringAlgos/dbScanClustering.js";
import * as gmm from "./clusteringAlgos/gmmClustering.js";
import * as hierarchicalClustering from "./clusteringAlgos/hClustering.js";
import { loadIrisDataset } from "./dataset.js";
// Example dataset (each row is a 2D point)
const data = await loadIrisDataset();

// Run K-Means with k=3
await kMeansClustering
  .kMeansClustering(data, 3)
  .then(({ assignments, centroids }) => {
    console.log("Cluster Assignments:", assignments);
    console.log("Centroids:", centroids);
  });

// const hierarchicalClusters = await hierarchicalClustering.hierarchicalClustering(data, 3);
// console.log("Final hierarchicalClusters: ", hierarchicalClusters);

const dbScanClusters = dbscan.dbscan(data, 0.5, 5);
console.log("DBSCAN Cluster Assignments:", dbScanClusters);

// const gmmClusters = await gmm.gmm(data, 3);
// console.log("GMM Cluster Assignments:", gmmClusters);
