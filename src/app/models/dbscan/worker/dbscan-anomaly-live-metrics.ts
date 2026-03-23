import type { DBSCANCallbackParameters, DBSCANParams, Model } from '@/ml/types';
import type { DBSCANAnomalyTrainingReport } from '../types';
import {
    getSafeMatrixFromTensor,
    type DatasetManager,
    type LiveMetrics,
} from '@/app/shared/workers';
import type { MatrixLike } from '@/app/shared/helpers';

export class DBSCANAnomalyLiveMetrics
    implements LiveMetrics<DBSCANCallbackParameters, DBSCANAnomalyTrainingReport>
{
    private model: Model<DBSCANParams>;
    private datasetManager: DatasetManager;

    static factory(model: Model<DBSCANParams>, datasetManager: DatasetManager) {
        return new DBSCANAnomalyLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<DBSCANParams>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    async calculateMetrics(params: DBSCANCallbackParameters): Promise<DBSCANAnomalyTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const { params: modelParams, numClusters, activePointIndex } = params;

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
            type: 'dbscan',
            taskType: 'anomaly',
            numClusters,
            activePointIndex,
            trainAnomalyRate: calcAnomalyRate(trainPredictions),
            testAnomalyRate: testPredictions ? calcAnomalyRate(testPredictions) : undefined,
            trainPredictions,
            testPredictions,
            params: modelParams ?? null,
        };
    }
}

function calcAnomalyRate(predictions: MatrixLike): number {
    const total = predictions.shape[0];
    if (total === 0) return 0;
    let anomalies = 0;
    for (let i = 0; i < total; i++) {
        if (predictions.array[i * predictions.shape[1]] === -1) anomalies++;
    }
    return anomalies / total;
}
