// Function to compute K-Means clustering
import * as tf from "@tensorflow/tfjs";
export async function kMeansClustering(data, k, maxIterations = 10) {
  const numSamples = data.length;

  // Convert data to TensorFlow tensor
  let points = tf.tensor2d(data);

  // Randomly initialize centroids
  let shuffledIndices = Array.from(
    tf.util.createShuffledIndices(numSamples).slice(0, k)
  );
  let centroids = points.gather(tf.tensor1d(shuffledIndices, "int32"));

  let assignments = new Array(numSamples).fill(0);

  for (let iter = 0; iter < maxIterations; iter++) {
    // Compute distances between points and centroids
    const expandedPoints = points.expandDims(1);
    const expandedCentroids = centroids.expandDims(0);
    const distances = tf.sum(
      tf.square(tf.sub(expandedPoints, expandedCentroids)),
      2
    );

    // Assign each point to the closest centroid
    assignments = distances.argMin(1).arraySync();

    // Update centroids based on mean of assigned points
    const newCentroids = [];
    for (let i = 0; i < k; i++) {
      const clusterIndices = assignments
        .map((label, idx) => (label === i ? idx : -1))
        .filter((idx) => idx !== -1);

      if (clusterIndices.length > 0) {
        // Convert clusterIndices to a proper tensor
        const clusterTensor = tf.tensor1d(clusterIndices, "int32");
        const clusterPoints = points.gather(clusterTensor);
        newCentroids.push(clusterPoints.mean(0));
      } else {
        newCentroids.push(centroids.gather(tf.tensor1d([i], "int32")));
      }
    }

    centroids = tf.stack(newCentroids);
  }

  return { assignments, centroids: centroids.arraySync() };
}
