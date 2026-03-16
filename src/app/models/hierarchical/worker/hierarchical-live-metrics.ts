import type {
    HierarchicalClusteringCallbackParameters,
    Model,
    HierarchicalClusteringParams,
} from '@/ml/types';
import type { HierarchicalClusteringTrainingReport } from '../types';
import type { DatasetManager, LiveMetrics } from '@/app/shared/workers';
import { getSafeMatrixFromTensor } from '@/app/shared/workers';
import type { MatrixLike } from '@/app/shared/helpers';

export class HierarchicalLiveMetrics
    implements
        LiveMetrics<HierarchicalClusteringCallbackParameters, HierarchicalClusteringTrainingReport>
{
    private model: Model<HierarchicalClusteringParams>;
    private datasetManager: DatasetManager;

    static factory(model: Model<HierarchicalClusteringParams>, datasetManager: DatasetManager) {
        return new HierarchicalLiveMetrics(model, datasetManager);
    }

    private constructor(
        model: Model<HierarchicalClusteringParams>,
        datasetManager: DatasetManager,
    ) {
        this.model = model;
        this.datasetManager = datasetManager;
    }

    async calculateMetrics(
        params: HierarchicalClusteringCallbackParameters,
    ): Promise<HierarchicalClusteringTrainingReport> {
        const { assignments, numClusters, params: modelParams } = params;

        const trainAssignments: MatrixLike = {
            array: assignments,
            shape: [assignments.length, 1],
        };

        let testAssignments: MatrixLike | undefined;
        const testData = this.datasetManager.getTestData();
        if (testData && modelParams) {
            const testAssignmentsTensor = this.model.predict(testData.X, modelParams);
            testAssignments = await getSafeMatrixFromTensor(testAssignmentsTensor);
            testAssignmentsTensor.dispose();
        }

        return {
            type: 'hierarchical',
            taskType: 'clustering',
            numClusters,
            trainAssignments,
            testAssignments,
            params: modelParams ?? null,
        };
    }
}
