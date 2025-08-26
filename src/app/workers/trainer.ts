import { createModel } from '@/app/helpers/createModel';
import type {
    ModelRepresentation,
    OptimizerCallbackParameters,
    TrainingEventEmitter,
    TrainingState,
} from '@/ml/types';
import type { PreprocessingModelDecorator } from '@/ml/models';
import type { State } from '@/app/store';
import { Tensor, memory } from '@tensorflow/tfjs';
import { DatasetManager } from './dataset-manager';
import { TrainingSession } from './training-session';
import { LiveMetricsProps } from './live-metrics-props';
import { LiveMetrics } from './live-metrics';
import { TrainingReportGenerator } from './training-report-generator';

type TrainingCallbacks = {
    onReport: (report: Float32Array) => void;
    onState: (state: TrainingState) => void;
    onInfo: (msg: string) => void;
    onError: (msg: string) => void;
    onFinished: () => void;
};

type TrainedModel = PreprocessingModelDecorator<ModelRepresentation>;

export class Trainer {
    private model: TrainedModel;
    private datasetManager: DatasetManager;
    private eventEmitter: TrainingEventEmitter<ModelRepresentation>;
    private callbacks: TrainingCallbacks;
    private liveMetrics: LiveMetrics;
    private liveMetricsProps: LiveMetricsProps;
    private reportGenerator: TrainingReportGenerator;

    private trainingSession: TrainingSession | null = null;
    private isTraining = false;
    private byStep = false;

    constructor(state: State, callbacks: TrainingCallbacks) {
        const { modelSettings, dataSettings, data } = state;

        const [model, eventEmitter] = createModel(modelSettings, dataSettings);

        this.model = model;
        this.callbacks = callbacks;
        this.eventEmitter = eventEmitter;
        this.datasetManager = new DatasetManager(data);
        this.liveMetricsProps = new LiveMetricsProps(state);
        this.liveMetrics = new LiveMetrics(model, this.datasetManager, this.liveMetricsProps);
        this.reportGenerator = new TrainingReportGenerator();

        // Set up event handling
        this.setupEventHandlers();
    }

    async train(byStep: boolean): Promise<void> {
        if (this.isTraining) {
            throw new Error('Training already in progress');
        }

        this.isTraining = true;

        console.info('Training started', memory());

        try {
            await this.executeTraining(byStep);
        } catch (error) {
            const prepareError = error instanceof Error ? error : new Error(String(error));
            this.handleTrainingError(prepareError);
        } finally {
            this.cleanup();
            this.isTraining = false;
        }

        console.info('Training finished', memory());
    }

    stop() {
        this.model.stop();
        this.isTraining = false;
    }

    pause() {
        this.model.pause();
    }

    resume() {
        this.model.resume();
    }

    step() {
        this.model.step();
    }

    private async executeTraining(byStep: boolean): Promise<void> {
        this.trainingSession = new TrainingSession(this.liveMetricsProps);
        this.byStep = byStep;

        const model = this.model;
        const datasetManager = this.datasetManager;
        const callbacks = this.callbacks;

        await this.trainModel(model, datasetManager);

        callbacks.onFinished?.();
    }

    private async trainModel(model: TrainedModel, datasetManager: DatasetManager): Promise<void> {
        const { X, y } = datasetManager.getTrainingData();

        console.time('Model Training');

        const theta = await model.train(X, y);

        console.timeEnd('Model Training');

        /**
         * Theta
         * Linear Regression
         * [
         *  [... bias ...],
         *  [... waight1 ...],
         *  [... waight2 ...]
         * ]
         *
         * Logistic Regression
         * [     cat1, cat2
         *  [... bias, bias ...],
         *  [... waight1, waight1 ...],
         *  [... waight2, waight2 ...]
         * ]
         *
         */

        if (theta instanceof Tensor) {
            theta.print();
            theta.dispose();
        }
    }

    private setupEventHandlers(): void {
        const eventEmitter = this.eventEmitter;
        const datasetManager = this.datasetManager;
        const callbacks = this.callbacks;

        eventEmitter.on('info', callbacks.onInfo);
        eventEmitter.on('error', (message) => {
            callbacks.onError(message);
            this.stop();
        });

        eventEmitter.on('state', (state) => {
            callbacks.onState(state);
            this.handleStateChange(state, datasetManager);
        });

        this.eventEmitter.on('callback', async (params) => {
            await this.handleTrainingIteration(params);
        });
    }

    private async handleTrainingIteration(
        params: OptimizerCallbackParameters<ModelRepresentation>,
    ): Promise<void> {
        if (!this.trainingSession) return;

        // Update training session state
        this.trainingSession.updateIteration(
            params.threadId,
            params.iteration,
            params.theta,
            params.loss,
        );

        const liveResults = await this.liveMetrics.calculate(this.trainingSession);

        const report = this.reportGenerator.generateReport(liveResults, this.trainingSession);

        this.callbacks.onReport(report);

        // Handle step-by-step learning mode
        if (this.shouldStopAfterIteration(params.iteration)) {
            this.model.pause();
        }
    }

    private shouldStopAfterIteration(iteration: number): boolean {
        return this.byStep && iteration === 0;
    }

    private handleStateChange(state: TrainingState, datasetManager: DatasetManager): void {
        if (state === 'transforming') {
            // Pre-cache transformed data
            const testData = datasetManager.getTestData();
            const predictionData = datasetManager.getPredictionData();

            if (this.model && testData) {
                this.model.prepareFeatures(testData.X);
                this.model.prepareLabels(testData.y);
            }

            if (this.model && predictionData) {
                this.model.prepareFeatures(predictionData);
            }
        }
    }

    private cleanup(): void {
        this.model.dispose(true);
        this.eventEmitter.clear();
        this.datasetManager.dispose();
        this.trainingSession?.dispose();
    }

    private handleTrainingError(error: Error): void {
        console.error('Training failed:', error);
        this.callbacks.onError(`Training failed: ${error.message}`);
    }
}
