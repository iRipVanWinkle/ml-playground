import * as tf from '@tensorflow/tfjs';

export type NormalizatorFn = (tensor: tf.Tensor2D) => tf.Tensor2D;

/**
 * Damy normalization to a tf.Tensor2D.
 *
 * @param tensor - A tf.Tensor2D to be z-score scaled.
 * @returns A new tf.Tensor2D with z-score normalized values.
 */
export function noScaling(tensor: tf.Tensor2D): tf.Tensor2D {
    return tensor;
}

export * from './zScoreScaling';
