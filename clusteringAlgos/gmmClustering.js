import * as tf from "@tensorflow/tfjs";
import * as math from "mathjs";
import { inverseMatrix } from "../utils/helpers.js";

// Function to compute Gaussian probability density function (PDF)
function gaussianPDF(x, mean, covariance) {
  const d = mean.shape[0];
  const det = math.det(covariance);
  const invCov = inverseMatrix(covariance);

  const factor = 1 / Math.sqrt((2 * Math.PI) ** d * det);
  const diff = tf.sub(x, mean);
  const exponent = tf
    .neg(tf.dot(tf.dot(diff, invCov), diff.transpose()))
    .div(2);

  return factor * tf.exp(exponent);
}

// GMM Function
export async function gmm(data, k, maxIterations = 100) {
  const n = data.length;
  const d = data[0].length;

  let means = tf.randomNormal([k, d]);
  let covariances = tf.eye(d).expandDims(0).tile([k, 1, 1]); // Identity matrices
  let weights = tf.div(tf.ones([k]), k); // Equal weight for each cluster

  let responsibilities = tf.zeros([n, k]);

  for (let iter = 0; iter < maxIterations; iter++) {
    // E-Step: Compute responsibilities
    let pdfs = data.map((point) => {
      return tf.stack(
        means
          .arraySync()
          .map((mean, j) =>
            gaussianPDF(
              tf.tensor1d(point),
              tf.tensor1d(mean),
              covariances.slice([j, 0, 0], [1, d, d]).squeeze()
            )
          )
      );
    });

    responsibilities = tf.stack(pdfs).mul(weights.expandDims(0));
    responsibilities = responsibilities.div(tf.sum(responsibilities, 1, true));

    // M-Step: Update means, covariances, and weights
    let Nk = tf.sum(responsibilities, 0);
    means = tf
      .matMul(responsibilities.transpose(), tf.tensor(data))
      .div(Nk.expandDims(1));

    let updatedCovariances = [];
    for (let j = 0; j < k; j++) {
      let diff = tf.sub(tf.tensor(data), means.slice([j, 0], [1, d]).squeeze());
      let weightedCov = tf.matMul(
        diff.transpose(),
        tf.mul(diff, responsibilities.slice([0, j], [-1, 1]))
      );
      updatedCovariances.push(weightedCov.div(Nk.slice([j], [1])));
    }
    covariances = tf.stack(updatedCovariances);
    weights = Nk.div(n);
  }

  // Assign clusters based on max responsibility
  let clusterAssignments = tf.argMax(responsibilities, 1).arraySync();
  return clusterAssignments;
}
