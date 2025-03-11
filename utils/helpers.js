// Function to compute Euclidean distance
import * as tf from "@tensorflow/tfjs";
export function euclideanDistance(a, b) {
  return tf.sqrt(tf.sum(tf.square(tf.sub(a, b)))).arraySync();
}
