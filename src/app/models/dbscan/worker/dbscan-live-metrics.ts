import type { DBSCANCallbackParameters, DBSCANParams, Model } from '@/ml/types';
import type { DBSCANTrainingReport } from '../types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import { getSafeMatrixFromTensor, getSafeTensorValue } from '@/app/shared/workers';
import type { MatrixLike } from '@/app/shared/helpers';
import { silhouetteScore } from '@/ml/metrics';
import { tensor2d } from '@tensorflow/tfjs';

export class DBSCANLiveMetrics
    implements LiveMetrics<DBSCANCallbackParameters, DBSCANTrainingReport>
{
    private model: Model<DBSCANParams>;
    private datasetManager: DatasetManager;

    static factory(model: Model<DBSCANParams>, datasetManager: DatasetManager) {
        return new DBSCANLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<DBSCANParams>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    async calculateMetrics(params: DBSCANCallbackParameters): Promise<DBSCANTrainingReport> {
        const { assignments, numClusters, activePointIndex, params: modelParams } = params;

        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();

        const trainAssignmentsArray: MatrixLike = {
            array: assignments,
            shape: [assignments.length, 1],
        };
        const trainAssignments = tensor2d(trainAssignmentsArray.array, trainAssignmentsArray.shape);

        const trainSilhouetteScore =
            numClusters >= 2
                ? silhouetteScore(trainingData.X, trainAssignments, numClusters)
                : undefined;

        let testAssignments;
        let testSilhouetteScore;
        if (testData) {
            testAssignments = this.model.predict(testData.X, modelParams);
            testSilhouetteScore =
                numClusters >= 2
                    ? silhouetteScore(testData.X, testAssignments, numClusters)
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
