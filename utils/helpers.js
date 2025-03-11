import * as tf from "@tensorflow/tfjs";
import * as math from "mathjs";

// Function to compute Euclidean distance
export function euclideanDistance(a, b) {
  return tf.sqrt(tf.sum(tf.square(tf.sub(a, b)))).arraySync();
}

export function inverseMatrix(matrix) {
  const values = matrix.arraySync(); // Convert Tensor to Array
  return math.inv(values); // Compute inverse using math.js
}
