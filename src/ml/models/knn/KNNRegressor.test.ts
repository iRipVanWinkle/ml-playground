import { tensor2d } from '@tensorflow/tfjs';
import { describe, expect, it } from 'vitest';
import { KNNRegressor } from './KNNRegressor';

describe('KNNRegressor', () => {
    it('predicts the mean of nearest training values', async () => {
        const XTrain = tensor2d([[0], [2], [4], [6], [8]]);
        const yTrain = tensor2d([[0], [2], [4], [6], [8]]);

        const model = new KNNRegressor({ k: 2 });
        await model.train(XTrain, yTrain);

        // Predict at x=3 → nearest are [2, 4] → mean = 3
        const XTest = tensor2d([[3]]);
        const predictions = model.predict(XTest);
        const result = (await predictions.array()) as number[][];

        expect(result[0][0]).toBeCloseTo(3, 1);

        predictions.dispose();
        model.dispose();
        XTrain.dispose();
        yTrain.dispose();
        XTest.dispose();
    });

    it('handles k=1 (exact nearest neighbor)', async () => {
        const XTrain = tensor2d([[0], [10]]);
        const yTrain = tensor2d([[5], [100]]);

        const model = new KNNRegressor({ k: 1 });
        await model.train(XTrain, yTrain);

        const XTest = tensor2d([[1]]);
        const predictions = model.predict(XTest);
        const result = (await predictions.array()) as number[][];

        expect(result[0][0]).toBeCloseTo(5, 3);

        predictions.dispose();
        model.dispose();
        XTrain.dispose();
        yTrain.dispose();
        XTest.dispose();
    });

    it('supports distance-weighted regression', async () => {
        // Closer point has much higher weight
        const XTrain = tensor2d([[0], [1], [100]]);
        const yTrain = tensor2d([[0], [1], [1000]]);

        const model = new KNNRegressor({ k: 3, weights: 'distance' });
        await model.train(XTrain, yTrain);

        // Test at x=0.01 — should be very close to y=0
        const XTest = tensor2d([[0.01]]);
        const predictions = model.predict(XTest);
        const result = (await predictions.array()) as number[][];

        expect(result[0][0]).toBeLessThan(5);

        predictions.dispose();
        model.dispose();
        XTrain.dispose();
        yTrain.dispose();
        XTest.dispose();
    });

    it('predictWithMetadata returns regression type', async () => {
        const XTrain = tensor2d([[0], [1]]);
        const yTrain = tensor2d([[0], [1]]);

        const model = new KNNRegressor({ k: 1 });
        await model.train(XTrain, yTrain);

        const XTest = tensor2d([[0.1]]);
        const meta = model.predictWithMetadata(XTest);

        expect(meta.type).toBe('regression');

        meta.dispose();
        model.dispose();
        XTrain.dispose();
        yTrain.dispose();
        XTest.dispose();
    });

    it('throws when model is not trained', () => {
        const model = new KNNRegressor({ k: 3 });
        const X = tensor2d([[1, 2]]);
        expect(() => model.predict(X)).toThrow();
        X.dispose();
    });
});
