import { type Tensor2D } from '@tensorflow/tfjs';
import type { KMeansCallbackParameters, Model, ModelRepresentation } from '@/ml/types';
import {
    getSafeMatrixFromTensor,
    type DatasetManager,
    type LiveMetrics,
} from '@/app/shared/workers';
import type { KMeansSettings, KMeansTrainingReport } from '../types';
import { kMeansMetricData } from '@/app/shared/visualization/metrics/k-means/calculations';
import type { TrainingSettings } from '../../types';
import { distanceFactory } from '@/ml/factories';
import type { DistanceMetric } from '@/ml/distance';

export class KMeansLiveMetrics
    implements LiveMetrics<KMeansCallbackParameters, KMeansTrainingReport>
{
    private inertiaHistory: number[] = [];

    private model: Model<ModelRepresentation>;
    private datasetManager: DatasetManager;
    private distanceMetric: DistanceMetric;

    static factory(
        model: Model<Tensor2D>,
        datasetManager: DatasetManager,
        settings: TrainingSettings<KMeansSettings>,
    ) {
        const distanceMetric = distanceFactory(settings.modelSettings.distance);
        return new KMeansLiveMetrics(model, datasetManager, distanceMetric);
    }

    private constructor(
        model: Model<Tensor2D>,
        datasetManager: DatasetManager,
        distanceMetric: DistanceMetric,
    ) {
        this.model = model;
        this.datasetManager = datasetManager;
        this.distanceMetric = distanceMetric;
    }

    async calculateMetrics(params: KMeansCallbackParameters): Promise<KMeansTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const distanceMetric = this.distanceMetric;

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
            kMeansMetricData(trainingData.X, assignments, centroids, distanceMetric),
            testData && testAssignments
                ? kMeansMetricData(testData.X, testAssignments, centroids, distanceMetric)
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
