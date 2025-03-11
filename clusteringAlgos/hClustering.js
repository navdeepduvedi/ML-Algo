import * as tf from "@tensorflow/tfjs";
import * as helpers from "../utils/helpers.js";
// Hierarchical Clustering function
export async function hierarchicalClustering(data, numClusters) {
  let clusters = data.map((point, index) => ({ id: index, points: [point] }));

  while (clusters.length > numClusters) {
    let minDist = Infinity;
    let mergeA = -1,
      mergeB = -1;

    // Find the two closest clusters
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const dist = helpers.euclideanDistance(
          tf.tensor1d(clusters[i].points.flat()),
          tf.tensor1d(clusters[j].points.flat())
        );

        if (dist < minDist) {
          minDist = dist;
          mergeA = i;
          mergeB = j;
        }
      }
    }

    // Merge the two closest clusters
    let newCluster = {
      id: clusters.length,
      points: [...clusters[mergeA].points, ...clusters[mergeB].points],
    };

    // Remove old clusters and add new one
    clusters = clusters.filter((_, idx) => idx !== mergeA && idx !== mergeB);
    clusters.push(newCluster);
  }

  return clusters.map((c, index) => ({
    cluster: index,
    points: c.points.length,
  }));
}
