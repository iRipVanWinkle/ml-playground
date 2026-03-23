import type { DBSCANCallbackParameters, DBSCANParams, Model } from '@/ml/types';
import type { DBSCANSettings, DBSCANTrainingReport } from '../types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import { getSafeMatrixFromTensor, getSafeTensorValue } from '@/app/shared/workers';
import type { MatrixLike } from '@/app/shared/helpers';
import { silhouetteScore } from '@/ml/metrics';
import { tensor2d } from '@tensorflow/tfjs';
import type { DistanceMetric } from '@/ml/distance';
import { distanceFactory } from '@/ml/factories';
import type { TrainingSettings } from '../../types';

export class DBSCANClusteringLiveMetrics
    implements LiveMetrics<DBSCANCallbackParameters, DBSCANTrainingReport>
{
    private model: Model<DBSCANParams>;
    private datasetManager: DatasetManager;
    private distanceMetric: DistanceMetric;

    static factory(
        model: Model<DBSCANParams>,
        datasetManager: DatasetManager,
        settings: TrainingSettings<DBSCANSettings>,
    ) {
        const distanceMetric = distanceFactory(settings.modelSettings.distance);
        return new DBSCANClusteringLiveMetrics(model, datasetManager, distanceMetric);
    }

    private constructor(
        model: Model<DBSCANParams>,
        datasetManager: DatasetManager,
        distanceMetric: DistanceMetric,
    ) {
        this.model = model;
        this.datasetManager = datasetManager;
        this.distanceMetric = distanceMetric;
    }

    async calculateMetrics(params: DBSCANCallbackParameters): Promise<DBSCANTrainingReport> {
        const { assignments, numClusters, activePointIndex, params: modelParams } = params;
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const distanceMetric = this.distanceMetric;

        const trainAssignmentsArray: MatrixLike = {
            array: assignments,
            shape: [assignments.length, 1],
        };
        const trainAssignments = tensor2d(trainAssignmentsArray.array, trainAssignmentsArray.shape);

        const trainSilhouetteScore =
            numClusters >= 2
                ? silhouetteScore(trainingData.X, trainAssignments, numClusters, distanceMetric)
                : undefined;

        let testAssignments;
        let testSilhouetteScore;
        if (testData) {
            testAssignments = this.model.predict(testData.X, modelParams);
            testSilhouetteScore =
                numClusters >= 2
                    ? silhouetteScore(testData.X, testAssignments, numClusters, distanceMetric)
                    : undefined;
        }

        const [testAssignmentsArray, trainSilhouetteScoreValue, testSilhouetteScoreValue] =
            await Promise.all([
                getSafeMatrixFromTensor(testAssignments),
                getSafeTensorValue(trainSilhouetteScore),
                getSafeTensorValue(testSilhouetteScore),
            ]);

        trainAssignments.dispose();
        testAssignments?.dispose();
        trainSilhouetteScore?.dispose();
        testSilhouetteScore?.dispose();

        return {
            type: 'dbscan',
            taskType: 'clustering',
            numClusters,
            activePointIndex,
            trainAssignments: trainAssignmentsArray,
            testAssignments: testAssignmentsArray,
            params: modelParams ?? null,
            trainSilhouetteScore: trainSilhouetteScoreValue,
            testSilhouetteScore: testSilhouetteScoreValue,
        };
    }
}
