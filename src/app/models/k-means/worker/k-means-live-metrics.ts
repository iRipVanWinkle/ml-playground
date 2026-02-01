import { type Tensor2D } from '@tensorflow/tfjs';
import type { KMeansCallbackParameters, Model, ModelRepresentation } from '@/ml/types';
import {
    getSafeMatrixFromTensor,
    type DatasetManager,
    type LiveMetrics,
} from '@/app/shared/workers';
import type { KMeansTrainingReport } from '../types';
import { kMeansMetricData } from '@/app/shared/visualization/metrics/k-means/calculations';

export class KMeansLiveMetrics
    implements LiveMetrics<KMeansCallbackParameters, KMeansTrainingReport>
{
    private inertiaHistory: number[] = [];

    private model: Model<ModelRepresentation>;
    private datasetManager: DatasetManager;

    static factory(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        return new KMeansLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    async calculateMetrics(params: KMeansCallbackParameters): Promise<KMeansTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();

        const { iteration, assignments, centroids, inertia } = params;

        this.inertiaHistory.push(inertia);

        let testAssignments;
        if (testData) {
            testAssignments = this.model.predict(testData.X, centroids);
        }

        const [
            trainAssignmentsArray,
            centroidsArray,
            testAssignmentsArray,
            trainMetrics,
            testMetrics,
        ] = await Promise.all([
            getSafeMatrixFromTensor(assignments),
            getSafeMatrixFromTensor(centroids),
            getSafeMatrixFromTensor(testAssignments),
            kMeansMetricData(trainingData.X, assignments, centroids),
            testData && testAssignments
                ? kMeansMetricData(testData.X, testAssignments, centroids)
                : undefined,
        ]);

        testAssignments?.dispose();

        return {
            type: 'k-means',
            taskType: 'clustering',
            iteration: iteration + 1,
            centroids: centroidsArray,
            trainAssignments: trainAssignmentsArray,
            testAssignments: testAssignmentsArray,
            trainMetrics,
            testMetrics,
            inertiaHistory: this.inertiaHistory,
        };
    }
}
