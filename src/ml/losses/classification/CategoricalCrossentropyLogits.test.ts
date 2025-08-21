import { tensor2d } from '@tensorflow/tfjs';
import { CategoricalCrossentropyLogits } from './CategoricalCrossentropyLogits';
import { beforeEach, describe, expect, it } from 'vitest';
import type { LossFunction } from '../../types';

describe('CategoricalCrossentropyLogits', () => {
    let loss: LossFunction;

    beforeEach(() => {
        loss = new CategoricalCrossentropyLogits();
    });

    describe('::compute', () => {
        it('should compute loss for perfect predictions', async () => {
            // Perfect prediction: high logit for correct class, low for others
            const yTrue = tensor2d([
                [1, 0, 0],
                [0, 1, 0],
            ]);
            const logits = tensor2d([
                [10, -10, -10],
                [-10, 10, -10],
            ]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should be very close to 0 for perfect predictions
            expect(value[0]).toBeCloseTo(0, 5);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should compute loss for completely wrong predictions', async () => {
            // Wrong prediction: high logit for wrong class
            const yTrue = tensor2d([
                [1, 0, 0],
                [0, 1, 0],
            ]);
            const logits = tensor2d([
                [-10, 10, -10],
                [10, -10, -10],
            ]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should be high loss for wrong predictions
            expect(value[0]).toBeGreaterThan(10);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should compute loss for uniform predictions', async () => {
            // Uniform logits (uncertain prediction)
            const yTrue = tensor2d([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]);
            const logits = tensor2d([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should be approximately log(3) ≈ 1.099 for 3-class uniform distribution
            expect(value[0]).toBeCloseTo(Math.log(3), 3);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should handle single sample', async () => {
            const yTrue = tensor2d([[0, 1, 0]]);
            const logits = tensor2d([[1, 2, 0]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Manual calculation: log(exp(1) + exp(2) + exp(0)) - 2
            // = log(2.718 + 7.389 + 1) - 2 = log(11.107) - 2 ≈ 2.407 - 2 = 0.407
            expect(value[0]).toBeCloseTo(0.4076, 3);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should handle large positive logits without overflow', async () => {
            // Test numerical stability with large logits
            const yTrue = tensor2d([[1, 0, 0]]);
            const logits = tensor2d([[100, 99, 98]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should not be NaN or Infinity
            expect(value[0]).toBeTypeOf('number');
            expect(value[0]).not.toBeNaN();

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should handle large negative logits without underflow', async () => {
            // Test numerical stability with large negative logits
            const yTrue = tensor2d([[1, 0, 0]]);
            const logits = tensor2d([[-100, -101, -102]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should not be NaN or Infinity
            expect(value[0]).toBeTypeOf('number');
            expect(value[0]).not.toBeNaN();

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should handle batch of samples correctly', async () => {
            const yTrue = tensor2d([
                [1, 0, 0], // Class 0
                [0, 1, 0], // Class 1
                [0, 0, 1], // Class 2
            ]);
            const logits = tensor2d([
                [2, 1, 0], // Prefers class 0 (correct)
                [0, 3, 1], // Prefers class 1 (correct)
                [1, 0, 4], // Prefers class 2 (correct)
            ]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // All predictions are correct, so loss should be relatively low
            expect(value[0]).toBeGreaterThan(0);
            expect(value[0]).toBeLessThan(2);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should be equivalent to manual categorical crossentropy calculation', async () => {
            const yTrue = tensor2d([
                [1, 0, 0],
                [0, 1, 0],
            ]);
            const logits = tensor2d([
                [1, 2, 3],
                [4, 5, 6],
            ]);

            const result = loss.compute(yTrue, logits);
            const lossValue = await result.data();

            // Manual calculation using the correct formula:
            // L = log(sum(exp(logits))) - sum(y_true * logits)

            // For sample 1: logits = [1, 2, 3], y_true = [1, 0, 0]
            // logSumExp = log(e^1 + e^2 + e^3) = log(2.718 + 7.389 + 20.086) = log(30.193) ≈ 3.407
            // weighted_logits = 1*1 + 0*2 + 0*3 = 1
            // loss1 = 3.407 - 1 = 2.407

            // For sample 2: logits = [4, 5, 6], y_true = [0, 1, 0]
            // logSumExp = log(e^4 + e^5 + e^6) = log(54.598 + 148.413 + 403.429) = log(606.44) ≈ 6.407
            // weighted_logits = 0*4 + 1*5 + 0*6 = 5
            // loss2 = 6.407 - 5 = 1.407

            // Wait, let me recalculate more precisely:
            // Sample 1: log(e^1 + e^2 + e^3) - 1 = log(2.718 + 7.389 + 20.086) - 1 ≈ 3.408 - 1 = 2.408
            // Sample 2: log(e^4 + e^5 + e^6) - 5 = log(54.598 + 148.413 + 403.429) - 5 ≈ 6.408 - 5 = 1.408
            // Mean loss = (2.408 + 1.408) / 2 = 1.908

            expect(lossValue[0]).toBeCloseTo(1.9076, 3);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should handle binary classification case', async () => {
            // Binary classification with 2 classes
            const yTrue = tensor2d([
                [1, 0],
                [0, 1],
            ]);
            const logits = tensor2d([
                [2, -1],
                [-2, 1],
            ]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            expect(value[0]).toBeTypeOf('number');
            expect(value[0]).toBeGreaterThan(0);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should handle edge case with zero logits', async () => {
            const yTrue = tensor2d([[1, 0, 0]]);
            const logits = tensor2d([[0, 0, 0]]);

            const result = loss.compute(yTrue, logits);
            const value = await result.data();

            // Should be log(3) ≈ 1.099 for uniform distribution
            expect(value[0]).toBeCloseTo(Math.log(3), 3);

            yTrue.dispose();
            logits.dispose();
            result.dispose();
        });

        it('should validate tensor shapes match', async () => {
            const yTrue = tensor2d([[1, 0, 0]]);
            const logits = tensor2d([[1, 2]]); // Wrong shape

            expect(() => {
                loss.compute(yTrue, logits);
            }).toThrow();

            yTrue.dispose();
            logits.dispose();
        });

        it('should be monotonic (lower loss for better predictions)', async () => {
            const yTrue = tensor2d([[1, 0, 0]]);

            // Good prediction (correct class has highest logit)
            const goodLogits = tensor2d([[5, 1, 1]]);
            const goodLoss = loss.compute(yTrue, goodLogits);
            const goodValue = await goodLoss.data();

            // Bad prediction (wrong class has highest logit)
            const badLogits = tensor2d([[1, 5, 1]]);
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
    });
});
