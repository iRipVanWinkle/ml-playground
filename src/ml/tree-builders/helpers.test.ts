import { describe, it, expect } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import {
    getColumnValues,
    splitIndices,
    findLeafNode,
    computeMeanValue,
    computeClassProbabilities,
    probabilityToClassIndex,
    bootstrapSample,
    subsampleFeatures,
    bootstrapFeatures,
} from './helpers';

describe('Tree Builder Helpers', () => {
    describe('getColumnValues', () => {
        it('should correctly extract values from a specific column based on indices', () => {
            const features = [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ];
            const indexes = [0, 2, 3];
            const columnIndex = 1;

            const result = getColumnValues(features, indexes, columnIndex);

            expect(result).toEqual([2, 8, 11]);
        });

        it('should return an empty array if indices are empty', () => {
            const features = [
                [1, 2, 3],
                [4, 5, 6],
            ];

            const result = getColumnValues(features, [], 0);

            expect(result).toEqual([]);
        });
    });

    describe('splitIndices', () => {
        it('should correctly split indices based on a threshold', () => {
            const featureValues = [10, 20, 30, 40, 50];
            const indices = [0, 1, 2, 3, 4];
            const threshold = 25;

            const result = splitIndices(featureValues, indices, threshold);

            expect(result.leftIndices).toEqual([0, 1]);
            expect(result.rightIndices).toEqual([2, 3, 4]);
        });

        it('should put all indices in left if all values are <= threshold', () => {
            const featureValues = [10, 20, 30];
            const indices = [1, 2, 3];
            const threshold = 50;

            const result = splitIndices(featureValues, indices, threshold);

            expect(result.leftIndices).toEqual([1, 2, 3]);
            expect(result.rightIndices).toEqual([]);
        });

        it('should put all indices in right if all values are > threshold', () => {
            const featureValues = [60, 70, 80];
            const indices = [4, 5, 6];
            const threshold = 50;

            const result = splitIndices(featureValues, indices, threshold);

            expect(result.leftIndices).toEqual([]);
            expect(result.rightIndices).toEqual([4, 5, 6]);
        });
    });

    describe('findLeafNode', () => {
        it('should correctly traverse a tree and return the leaf node', () => {
            const leaf1 = {
                featureIndex: null,
                threshold: null,
                leftChild: null,
                rightChild: null,
                value: 0,
            };
            const leaf2 = {
                featureIndex: null,
                threshold: null,
                leftChild: null,
                rightChild: null,
                value: 1,
            };
            const rootNode = {
                featureIndex: 0,
                threshold: 5,
                leftChild: leaf1,
                rightChild: leaf2,
                value: 0,
            };

            const sampleFeaturesLeft = [2];
            const sampleFeaturesRight = [7];

            expect(findLeafNode(sampleFeaturesLeft, rootNode)).toBe(leaf1);
            expect(findLeafNode(sampleFeaturesRight, rootNode)).toBe(leaf2);
        });
    });

    describe('computeMeanValue', () => {
        it('should correctly compute mean from target samples', () => {
            const targets = [[1], [3], [5]];
            const result = computeMeanValue(targets);
            expect(result.value).toBe(3);
        });
    });

    describe('computeClassProbabilities', () => {
        it('should compute correct class probabilities and highest value class', () => {
            const targets = [
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
            ];

            const result = computeClassProbabilities(targets);
            expect(result.probabilities).toEqual([0.25, 0.5, 0.25]);
            expect(result.value).toBe(1);
        });
    });

    describe('probabilityToClassIndex', () => {
        it('should correctly convert probability tensor to class indices', () => {
            tf.tidy(() => {
                const probability = tf.tensor2d([
                    [0.1, 0.8, 0.1],
                    [0.9, 0.05, 0.05],
                    [0.2, 0.3, 0.5],
                ]);

                const classIndices = probabilityToClassIndex(probability);
                expect(classIndices.shape).toEqual([3, 1]);
                expect(classIndices.arraySync()).toEqual([[1], [0], [2]]);
            });
        });
    });

    describe('bootstrapSample', () => {
        it('should create a bootstrapped sample of correct shape', () => {
            tf.tidy(() => {
                const features = tf.tensor2d([
                    [1, 2],
                    [3, 4],
                    [5, 6],
                ]);
                const targets = tf.tensor2d([[0], [1], [0]]);
                const seed = 42;

                const [bootstrappedFeatures, bootstrappedTargets] = bootstrapSample(
                    features,
                    targets,
                    seed,
                );

                expect(bootstrappedFeatures.shape).toEqual([3, 2]);
                expect(bootstrappedTargets.shape).toEqual([3, 1]);
            });
        });
    });

    describe('subsampleFeatures', () => {
        it('should create a subsampled feature matrix of correct size', () => {
            tf.tidy(() => {
                const features = tf.tensor2d([
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8],
                ]);
                const sampleSize = 2;
                const seed = 42;

                const subsampledFeatures = subsampleFeatures(features, sampleSize, seed);

                expect(subsampledFeatures.shape).toEqual([2, 2]);
                const subsampledArray = subsampledFeatures.arraySync();
                const originalArray = features.arraySync();

                subsampledArray.forEach((row: number[]) => {
                    expect(
                        originalArray.some(
                            (origRow: number[]) => origRow[0] === row[0] && origRow[1] === row[1],
                        ),
                    ).toBe(true);
                });
            });
        });
    });

    describe('bootstrapFeatures', () => {
        it('should create bootstrapped features with replacement', () => {
            tf.tidy(() => {
                const features = tf.tensor2d([
                    [1, 2],
                    [3, 4],
                    [5, 6],
                ]);
                const sampleSize = 5;
                const seed = 42;

                const bootstrapped = bootstrapFeatures(features, sampleSize, seed);

                expect(bootstrapped.shape).toEqual([3, 2]);
            });
        });
    });
});
