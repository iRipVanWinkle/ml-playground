import { tensor2d } from '@tensorflow/tfjs';
import { describe, expect, it } from 'vitest';
import { KNNClassifier } from './KNNClassifier';
import type { ClassificationMetadata } from '@/ml/types';

describe('KNNClassifier', () => {
    it('classifies a simple linearly separable dataset', async () => {
        const XTrain = tensor2d([
            [1, 1],
            [1, 2],
            [2, 1],
            [5, 5],
            [5, 6],
            [6, 5],
        ]);
        const yTrain = tensor2d([[0], [0], [0], [1], [1], [1]]);

        const model = new KNNClassifier({ k: 3 });
        await model.train(XTrain, yTrain);

        const XTest = tensor2d([
            [1.5, 1.5],
            [5.5, 5.5],
        ]);
        const predictions = model.predict(XTest);
        const result = (await predictions.array()) as number[][];

        expect(result[0][0]).toBe(0);
        expect(result[1][0]).toBe(1);

        predictions.dispose();
        model.dispose();
        XTrain.dispose();
        yTrain.dispose();
        XTest.dispose();
    });

    it('handles k=1 (nearest neighbor)', async () => {
        const XTrain = tensor2d([
            [0, 0],
            [10, 10],
        ]);
        const yTrain = tensor2d([[0], [1]]);

        const model = new KNNClassifier({ k: 1 });
        await model.train(XTrain, yTrain);

        const XTest = tensor2d([
            [0.5, 0.5],
            [9.5, 9.5],
        ]);
        const predictions = model.predict(XTest);
        const result = (await predictions.array()) as number[][];

        expect(result[0][0]).toBe(0);
        expect(result[1][0]).toBe(1);

        predictions.dispose();
        model.dispose();
        XTrain.dispose();
        yTrain.dispose();
        XTest.dispose();
    });

    it('predictWithMetadata returns classification type with probabilities', async () => {
        const XTrain = tensor2d([
            [0, 0],
            [1, 1],
            [10, 10],
            [11, 11],
        ]);
        const yTrain = tensor2d([[0], [0], [1], [1]]);

        const model = new KNNClassifier({ k: 2 });
        await model.train(XTrain, yTrain);

        const XTest = tensor2d([[0.5, 0.5]]);
        const meta = model.predictWithMetadata(XTest) as ClassificationMetadata;

        expect(meta.type).toBe('classification');
        const preds = (await meta.predictions.array()) as number[][];
        expect(preds[0][0]).toBe(0);

        meta.dispose();
        model.dispose();
        XTrain.dispose();
        yTrain.dispose();
        XTest.dispose();
    });

    it('supports distance-weighted voting', async () => {
        const XTrain = tensor2d([
            [0, 0],
            [1, 0],
            [10, 0],
        ]);
        const yTrain = tensor2d([[0], [0], [1]]);

        const model = new KNNClassifier({ k: 3, weights: 'distance' });
        await model.train(XTrain, yTrain);

        // Point very close to class-0 cluster
        const XTest = tensor2d([[0.1, 0]]);
        const predictions = model.predict(XTest);
        const result = (await predictions.array()) as number[][];

        expect(result[0][0]).toBe(0);

        predictions.dispose();
        model.dispose();
        XTrain.dispose();
        yTrain.dispose();
        XTest.dispose();
    });

    it('throws when model is not trained', () => {
        const model = new KNNClassifier({ k: 3 });
        const X = tensor2d([[1, 2]]);
        expect(() => model.predict(X)).toThrow();
        X.dispose();
    });
});
