import type {
    GaussianDistributionCallbackParameters,
    GaussianDistributionParams,
    Model,
} from '@/ml/types';
import type { GaussianDistributionTrainingReport } from '../types';
import {
    getSafeMatrixFromTensor,
    type DatasetManager,
    type LiveMetrics,
} from '@/app/shared/workers';
import type { MatrixLike } from '@/app/shared/helpers';

export class GaussianDistributionLiveMetrics
    implements
        LiveMetrics<GaussianDistributionCallbackParameters, GaussianDistributionTrainingReport>
{
    private model: Model<GaussianDistributionParams>;
    private datasetManager: DatasetManager;

    static factory(model: Model<GaussianDistributionParams>, datasetManager: DatasetManager) {
        return new GaussianDistributionLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<GaussianDistributionParams>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    async calculateMetrics(
        params: GaussianDistributionCallbackParameters,
    ): Promise<GaussianDistributionTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const { params: modelParams } = params;

        const trainPredictionsTensor = this.model.predict(trainingData.X, modelParams);
        const testPredictionsTensor = testData
            ? this.model.predict(testData.X, modelParams)
            : undefined;

        const [trainPredictions, testPredictions] = await Promise.all([
            getSafeMatrixFromTensor(trainPredictionsTensor),
            getSafeMatrixFromTensor(testPredictionsTensor),
        ]);

        trainPredictionsTensor.dispose();
        testPredictionsTensor?.dispose();

        return {
            type: 'gaussian-distribution',
            taskType: 'anomaly',
            trainAnomalyRate: calcAnomalyRate(trainPredictions),
            testAnomalyRate: testPredictions ? calcAnomalyRate(testPredictions) : undefined,
            trainPredictions,
            testPredictions,
            params: modelParams,
        };
    }
}

function calcAnomalyRate(predictions: MatrixLike): number {
    const total = predictions.shape[0];
    if (total === 0) return 0;
    let anomalies = 0;
    for (let i = 0; i < total; i++) {
        if (predictions.array[i * predictions.shape[1]] === 1) anomalies++;
    }
    return anomalies / total;
}
