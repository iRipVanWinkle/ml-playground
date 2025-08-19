import { describe, it, expect, beforeEach } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import type { LossFunction } from '../../types';
import { BinaryCrossentropyLogits } from './BinaryCrossentropyLogits';

describe('BinaryCrossentropyLogits', () => {
    let loss: LossFunction;

    beforeEach(() => {
        loss = new BinaryCrossentropyLogits();
    });

    describe('compute', () => {
        it('should compute loss for perfect positive predictions', async () => {
            // Perfect prediction: high positive logit for positive class
            const yTrue = tf.tensor2d([[1], [1]]);
            const logits = tf.tensor2d([[10], [10]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should be very close to 0 for perfect predictions
            expect(value[0]).toBeCloseTo(0, 4);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should compute loss for perfect negative predictions', async () => {
            // Perfect prediction: high negative logit for negative class
            const yTrue = tf.tensor2d([[0], [0]]);
            const logits = tf.tensor2d([[-10], [-10]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should be very close to 0 for perfect predictions
            expect(value[0]).toBeCloseTo(0, 4);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should compute loss for completely wrong predictions', async () => {
            // Wrong prediction: negative logit for positive class
            const yTrue = tf.tensor2d([[1], [1]]);
            const logits = tf.tensor2d([[-10], [-10]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should be high loss for wrong predictions
            expect(value[0]).toBeGreaterThan(5);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should compute loss for uncertain predictions (zero logits)', async () => {
            // Uncertain prediction: zero logits (50% probability)
            const yTrue = tf.tensor2d([[1], [0]]);
            const logits = tf.tensor2d([[0], [0]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should be log(2) ≈ 0.693 for uncertain predictions
            expect(value[0]).toBeCloseTo(Math.log(2), 3);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should handle single sample correctly', async () => {
            const yTrue = tf.tensor2d([[1]]);
            const logits = tf.tensor2d([[2]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Manual calculation: max(2, 0) - 2*1 + log(1 + exp(-2))
            // = 2 - 2 + log(1 + 0.135) = 0 + log(1.135) ≈ 0.127
            expect(value[0]).toBeCloseTo(0.1269, 3);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should handle large positive logits without overflow', async () => {
            // Test numerical stability with large positive logits
            const yTrue = tf.tensor2d([[1]]);
            const logits = tf.tensor2d([[100]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should not be NaN or Infinity
            expect(value[0]).toBeTypeOf('number');
            expect(value[0]).not.toBeNaN();
            expect(Number.isFinite(value[0])).toBe(true);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should handle large negative logits without underflow', async () => {
            // Test numerical stability with large negative logits
            const yTrue = tf.tensor2d([[0]]);
            const logits = tf.tensor2d([[-100]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should not be NaN or Infinity
            expect(value[0]).toBeTypeOf('number');
            expect(value[0]).not.toBeNaN();
            expect(Number.isFinite(value[0])).toBe(true);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should handle batch of mixed samples correctly', async () => {
            const yTrue = tf.tensor2d([[1], [0], [1], [0]]);
            const logits = tf.tensor2d([[2], [-1], [3], [-2]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // All predictions are correct, so loss should be relatively low
            expect(value[0]).toBeGreaterThan(0);
            expect(value[0]).toBeLessThan(1);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should be equivalent to manual binary crossentropy calculation', async () => {
            const yTrue = tf.tensor2d([[1], [0]]);
            const logits = tf.tensor2d([[1], [-1]]);

            const result = loss.compute(yTrue, logits);
            const lossValue = await result.data();

            // Manual calculation using stable formula:
            // Formula: max(logits, 0) - logits * y_true + log(1 + exp(-|logits|))
            // Sample 1: y=1, z=1
            // max(1, 0) - 1*1 + log(1 + exp(-|1|)) = 1 - 1 + log(1 + exp(-1)) = 0 + log(1.368) ≈ 0.313
            // Sample 2: y=0, z=-1
            // max(-1, 0) - (-1)*0 + log(1 + exp(-|-1|)) = 0 - 0 + log(1 + exp(-1)) = log(1.368) ≈ 0.313
            // Mean = (0.313 + 0.313) / 2 = 0.313

            expect(lossValue[0]).toBeCloseTo(0.3132, 3);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should handle edge case with extreme logits', async () => {
            const yTrue = tf.tensor2d([[1], [0]]);
            const logits = tf.tensor2d([[1000], [-1000]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should be very close to 0 for extreme correct predictions
            expect(value[0]).toBeCloseTo(0, 5);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should be monotonic (lower loss for better predictions)', async () => {
            const yTrue = tf.tensor2d([[1]]);

            // Good prediction (high positive logit for positive class)
            const goodLogits = tf.tensor2d([[5]]);
            const goodLoss = loss.compute(yTrue, goodLogits);
            const goodValue = await goodLoss.data();

            // Bad prediction (negative logit for positive class)
            const badLogits = tf.tensor2d([[-5]]);
            const badLoss = loss.compute(yTrue, badLogits);
            const badValue = await badLoss.data();

            // Good prediction should have lower loss
            expect(goodValue[0]).toBeLessThan(badValue[0]);

            yTrue.dispose();
            goodLogits.dispose();
            badLogits.dispose();
            goodLoss.dispose();
            badLoss.dispose();
        });

        it('should handle multi-column input (multiple features)', async () => {
            // Test with multiple samples and single output
            const yTrue = tf.tensor2d([[1], [0], [1]]);
            const logits = tf.tensor2d([[2], [-1], [3]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            expect(value[0]).toBeTypeOf('number');
            expect(Number.isFinite(value[0])).toBe(true);
            expect(value[0]).toBeGreaterThan(0);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should handle symmetric cases correctly', async () => {
            // Test symmetry: loss(y=1, z=x) should equal loss(y=0, z=-x)
            const yTrue1 = tf.tensor2d([[1]]);
            const logits1 = tf.tensor2d([[2]]);
            const loss1 = loss.compute(yTrue1, logits1);
            const value1 = await loss1.data();

            const yTrue2 = tf.tensor2d([[0]]);
            const logits2 = tf.tensor2d([[-2]]);
            const loss2 = loss.compute(yTrue2, logits2);
            const value2 = await loss2.data();

            expect(value1[0]).toBeCloseTo(value2[0], 5);

            yTrue1.dispose();
            logits1.dispose();
            yTrue2.dispose();
            logits2.dispose();
            loss1.dispose();
            loss2.dispose();
        });

        it('should be convex (loss increases as prediction gets worse)', async () => {
            const yTrue = tf.tensor2d([[1]]);

            // Test points along the logit axis
            const logits1 = tf.tensor2d([[2]]); // Good prediction
            const logits2 = tf.tensor2d([[0]]); // Neutral prediction
            const logits3 = tf.tensor2d([[-2]]); // Bad prediction

            const loss1 = loss.compute(yTrue, logits1);
            const loss2 = loss.compute(yTrue, logits2);
            const loss3 = loss.compute(yTrue, logits3);

            const value1 = await loss1.data();
            const value2 = await loss2.data();
            const value3 = await loss3.data();

            // Loss should increase as prediction gets worse
            expect(value1[0]).toBeLessThan(value2[0]);
            expect(value2[0]).toBeLessThan(value3[0]);

            yTrue.dispose();
            logits1.dispose();
            logits2.dispose();
            logits3.dispose();
            loss1.dispose();
            loss2.dispose();
            loss3.dispose();
        });
    });
});
