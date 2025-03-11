import * as tf from "@tensorflow/tfjs";
import * as helpers from "../utils/helpers.js";

// DBSCAN Clustering Algorithm
export function dbscan(data, eps, minPts) {
  const labels = new Array(data.length).fill(-1); // -1 means unclassified
  let clusterId = 0;

  function regionQuery(pointIndex) {
    return data
      .map((point, i) =>
        helpers.euclideanDistance(
          tf.tensor1d(data[pointIndex]),
          tf.tensor1d(point)
        ) < eps
          ? i
          : -1
      )
      .filter((i) => i !== -1);
  }

  function expandCluster(pointIndex, neighbors) {
    labels[pointIndex] = clusterId;

    let i = 0;
    while (i < neighbors.length) {
      let neighborIndex = neighbors[i];
      if (labels[neighborIndex] === -1) {
        labels[neighborIndex] = clusterId;
        let newNeighbors = regionQuery(neighborIndex);
        if (newNeighbors.length >= minPts) {
          neighbors = [...neighbors, ...newNeighbors];
        }
      }
      i++;
    }
  }

  for (let i = 0; i < data.length; i++) {
    if (labels[i] !== -1) continue;
    let neighbors = regionQuery(i);
    if (neighbors.length < minPts) {
      labels[i] = -1; // Mark as noise
    } else {
      expandCluster(i, neighbors);
      clusterId++;
    }
  }

  return labels;
}
