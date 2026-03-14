import type { IsolationEnsembleTree, Model, IsolationForestCallbackParameters } from '@/ml/types';
import type { IsolationForestTrainingReport } from '../types';
import {
    getSafeMatrixFromTensor,
    type DatasetManager,
    type LiveMetrics,
} from '@/app/shared/workers';
import { type MatrixLike } from '@/app/shared/helpers';

export class IsolationForestLiveMetrics
    implements LiveMetrics<IsolationForestCallbackParameters, IsolationForestTrainingReport>
{
    private model: Model<IsolationEnsembleTree>;
    private datasetManager: DatasetManager;

    static factory(model: Model<IsolationEnsembleTree>, datasetManager: DatasetManager) {
        return new IsolationForestLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<IsolationEnsembleTree>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    async calculateMetrics(
        params: IsolationForestCallbackParameters,
    ): Promise<IsolationForestTrainingReport> {
        const { ensemble } = params;

        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();

        const trainPredictionsTensor = this.model.predict(trainingData.X, ensemble);
        const testPredictionsTensor = testData
            ? this.model.predict(testData.X, ensemble)
            : undefined;

        const [trainPredictions, testPredictions] = await Promise.all([
            getSafeMatrixFromTensor(trainPredictionsTensor),
            getSafeMatrixFromTensor(testPredictionsTensor),
        ]);

        trainPredictionsTensor.dispose();
        testPredictionsTensor?.dispose();

        return {
            type: 'isolation-forest',
            taskType: 'anomaly',
            trainAnomalyRate: calcAnomalyRate(trainPredictions),
            testAnomalyRate: testPredictions ? calcAnomalyRate(testPredictions) : undefined,
            scoreThreshold: ensemble.scoreThreshold,
            trainPredictions,
            testPredictions,
            params: ensemble.trees,
        };
    }
}

function calcAnomalyRate(predictions: MatrixLike): number {
    const total = predictions.shape[0];
    if (total === 0) return 0;
    let anomalies = 0;
    for (let i = 0; i < total; i++) {
        // IsolationForest.predict returns -1 for anomalies
        if (predictions.array[i * predictions.shape[1]] === -1) anomalies++;
    }
    return anomalies / total;
}
