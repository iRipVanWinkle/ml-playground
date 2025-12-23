import type { Tensor2D } from '@tensorflow/tfjs';
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
    private iterationCount: number = 0;

    private model: Model<ModelRepresentation>;
    private datasetManager: DatasetManager;

    private assignments?: Tensor2D;
    private centroids?: Tensor2D;

    static factory(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        return new KMeansLiveMetrics(model, datasetManager);
    }

    private constructor(model: Model<Tensor2D>, datasetManager: DatasetManager) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    updateIteration(params: KMeansCallbackParameters): void {
        const { iteration, assignments, centroids, inertia } = params;

        this.assignments?.dispose();

        this.inertiaHistory.push(inertia);
        this.iterationCount = iteration + 1;
        this.assignments = assignments;
        this.centroids = centroids;
    }

    async calculateMetrics(): Promise<KMeansTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();

        let testAssignments;

        if (testData) {
            testAssignments = this.model.predict(testData.X, this.centroids);
        }

        const [
            trainAssignmentsArray,
            centroidsArray,
            testAssignmentsArray,
            trainMetrics,
            testMetrics,
        ] = await Promise.all([
            getSafeMatrixFromTensor(this.assignments!),
            getSafeMatrixFromTensor(this.centroids!),
            getSafeMatrixFromTensor(testAssignments),
            kMeansMetricData(trainingData.X, this.assignments!, this.centroids!),
            testData && testAssignments
                ? kMeansMetricData(testData.X, testAssignments, this.centroids!)
                : undefined,
        ]);

        return {
            type: 'k-means',
            taskType: 'clustering',
            iteration: this.iterationCount,
            centroids: centroidsArray,
            trainAssignments: trainAssignmentsArray,
            testAssignments: testAssignmentsArray,
            trainMetrics,
            testMetrics,
            inertiaHistory: this.inertiaHistory,
        };
    }

    dispose(): void {
        this.centroids?.dispose();
        this.assignments?.dispose();
    }
}
