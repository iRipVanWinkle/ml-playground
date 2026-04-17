import type { Scalar, Tensor2D } from '@tensorflow/tfjs';
import type { Model, TreeCallbackParameters, EnsembleTree, TreeNode } from '@/ml/types';
import {
    createTensorContainer,
    getSafeMatrixFromTensor,
    getSafeTensorValue,
    type DatasetManager,
    type LiveMetrics,
    type TensorContainer,
} from '@/app/shared/workers';
import {
    meanAbsoluteError,
    meanSquaredError,
    r2Score,
    residuals,
    rootMeanSquaredError,
} from '@/ml/metrics';
import type { TreeRegressionTrainingReport, TreeSettings } from '../types';
import type { TrainingSettings } from '../../types';
import { criterionFactory } from '@/ml/factories';

type MetricsTensors = {
    y: Tensor2D;
    loss: Scalar;
    mae: Scalar;
    mse: Scalar;
    rmse: Scalar;
    r2: Scalar;
    residuals: Tensor2D;
};

export class TreeRegressionLiveMetrics
    implements LiveMetrics<TreeCallbackParameters, TreeRegressionTrainingReport>
{
    private model: Model<EnsembleTree>;
    private datasetManager: DatasetManager;
    private modelSettings: TreeSettings;

    private trees: TreeNode[] = [];

    static factory(
        model: Model<EnsembleTree>,
        datasetManager: DatasetManager,
        settings: TrainingSettings,
    ) {
        const { modelSettings } = settings;

        if (modelSettings.type !== 'tree') {
            throw new Error(`Invalid settings type: expected 'tree', got '${modelSettings.type}'`);
        }

        return new TreeRegressionLiveMetrics(modelSettings, model, datasetManager);
    }

    private constructor(
        modelSettings: TreeSettings,
        model: Model<EnsembleTree>,
        datasetManager: DatasetManager,
    ) {
        this.model = model;
        this.datasetManager = datasetManager;
        this.modelSettings = modelSettings;
    }

    async calculateMetrics(params: TreeCallbackParameters): Promise<TreeRegressionTrainingReport> {
        const trainingData = this.datasetManager.getTrainingData();
        const testData = this.datasetManager.getTestData();
        const predictionData = this.datasetManager.getPredictionData();

        const { threadId, tree } = params;

        this.trees[threadId] = tree;

        let yPredictions: Tensor2D | undefined;
        if (predictionData) {
            yPredictions = this.model.predict(predictionData, this.trees);
        }

        const train = this.evaluateMetrics(trainingData.X, trainingData.y, this.trees);
        const test = testData
            ? this.evaluateMetrics(testData.X, testData.y, this.trees)
            : createTensorContainer<MetricsTensors, 'partial'>();

        const [
            predictionPredictedLabels,
            // train
            trainPredictedLabels,
            trainLossValue,
            trainMaeValue,
            trainMseValue,
            trainRmseValue,
            trainR2Value,
            trainResidualsArray,
            // test
            testPredictedLabels,
            testLossValue,
            testMaeValue,
            testMseValue,
            testRmseValue,
            testR2Value,
            testResidualsArray,
        ] = await Promise.all([
            getSafeMatrixFromTensor(yPredictions),
            // train
            getSafeMatrixFromTensor(train.y),
            getSafeTensorValue(train.loss),
            getSafeTensorValue(train.mae),
            getSafeTensorValue(train.mse),
            getSafeTensorValue(train.rmse),
            getSafeTensorValue(train.r2),
            getSafeMatrixFromTensor(train.residuals),
            // test
            getSafeMatrixFromTensor(test.y),
            getSafeTensorValue(test.loss),
            getSafeTensorValue(test.mae),
            getSafeTensorValue(test.mse),
            getSafeTensorValue(test.rmse),
            getSafeTensorValue(test.r2),
            getSafeMatrixFromTensor(test.residuals),
        ]);

        // Dispose of all tensors to free up memory
        yPredictions?.dispose();
        train.dispose();
        test.dispose();

        const hasTestMetrics =
            testMaeValue !== undefined &&
            testMseValue !== undefined &&
            testRmseValue !== undefined &&
            testR2Value !== undefined;

        return {
            type: 'tree',
            taskType: 'regression',
            trainLoss: trainLossValue,
            testLoss: testLossValue,
            trainPredictedLabels: trainPredictedLabels,
            testPredictedLabels: testPredictedLabels,
            predictionPredictedLabels: predictionPredictedLabels,
            params: this.trees,
            trainMetrics: {
                mae: trainMaeValue,
                mse: trainMseValue,
                rmse: trainRmseValue,
                r2: trainR2Value,
            },
            testMetrics: hasTestMetrics
                ? {
                      mae: testMaeValue,
                      mse: testMseValue,
                      rmse: testRmseValue,
                      r2: testR2Value,
                  }
                : undefined,
            trainResiduals: trainResidualsArray,
            testResiduals: testResidualsArray,
        };
    }

    private evaluateMetrics(
        X: Tensor2D,
        yTrue: Tensor2D,
        trees: EnsembleTree,
    ): TensorContainer<MetricsTensors> {
        const metrics = createTensorContainer<MetricsTensors>();

        metrics.y = this.model.predict(X, trees);
        metrics.mae = meanAbsoluteError(yTrue, metrics.y);
        metrics.mse = meanSquaredError(yTrue, metrics.y);
        metrics.rmse = rootMeanSquaredError(yTrue, metrics.y);
        metrics.r2 = r2Score(yTrue, metrics.y);
        metrics.residuals = residuals(yTrue, metrics.y);

        const criterion = criterionFactory(this.modelSettings.criterion);
        metrics.loss = criterion.loss(yTrue, metrics.y);

        return metrics;
    }
}
