import { Tensor, memory, ready, setBackend } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';

import type {
    CallbackParameters,
    Model,
    ModelRepresentation,
    TrainingControl,
    TrainingEventEmitter,
    TrainingState,
} from '@/ml/types';
import { PreprocessingModelDecorator } from '@/ml/models';
import type { Dataset, TaskType } from '@/app/shared/types';
import { DatasetManager } from '../../../../shared/workers/DatasetManager';
import { createPreprocessingPipeline } from '../../helpers';
import type { SystemSettings } from '@/app/features/configure-system';
import type { TransformationSettings } from '@/app/features/transform-data';
import type { ModelSettings, TrainingReport } from '@/app/models/types';
import { getWorkerRegistry } from '@/app/models/worker-registry';
import type { TrainingSettings } from '@/app/models/types';
import { EventEmitter } from '@/ml/events/EventEmitter';
import { TrainingController } from '@/ml/controllers/TrainingController';
import type { LiveMetrics } from '@/app/shared/workers';
import { Randomizer } from '@/ml/random/Randomizer';

type Settings = {
    taskType: TaskType;
    systemSettings: SystemSettings;
    dataSettings: TransformationSettings;
    data: Dataset;
    modelSettings: ModelSettings;
};

type TrainingCallbacks = {
    onReport: (report: TrainingReport) => void;
    onState: (state: TrainingState) => void;
    onInfo: (msg: string) => void;
    onError: (msg: string) => void;
    onFinished: () => void;
};

type TrainedModel = PreprocessingModelDecorator<ModelRepresentation>;

const workerRegistry = getWorkerRegistry();

export class TrainingOrchestrator {
    private model: TrainedModel;
    private datasetManager: DatasetManager;
    private trainingEventEmitter: TrainingEventEmitter;
    private trainingController: TrainingControl;
    private callbacks: TrainingCallbacks;
    private liveMetrics: LiveMetrics<CallbackParameters, TrainingReport>;

    private isTraining = false;
    private byStep = false;

    static async createOrchestrator(
        settings: Settings,
        callbacks: TrainingCallbacks,
    ): Promise<TrainingOrchestrator> {
        const { systemSettings } = settings;

        if (systemSettings.backend !== 'auto') {
            const wasmPath =
                import.meta.env.PROD && import.meta.env.BASE_URL !== '/'
                    ? `${import.meta.env.BASE_URL}wasm/`
                    : '/wasm/';

            setWasmPaths(wasmPath);
            setBackend(systemSettings.backend);
        }

        await ready();

        return new TrainingOrchestrator(settings, callbacks);
    }

    private constructor(settings: TrainingSettings, callbacks: TrainingCallbacks) {
        const { systemSettings, data } = settings;

        [this.model, this.trainingEventEmitter, this.trainingController] =
            this.createModel(settings);

        this.datasetManager = new DatasetManager(data);
        this.liveMetrics = this.createLiveMetrics(settings, this.model, this.datasetManager);

        this.callbacks = callbacks;

        Randomizer.setSeed(systemSettings.randomSeed);
        // Set up event handling
        this.setupEventHandlers();
    }

    async train(byStep: boolean): Promise<void> {
        if (this.isTraining) {
            throw new Error('Training already in progress');
        }

        this.byStep = byStep;
        this.isTraining = true;

        console.info('Training started', memory());

        try {
            await this.executeTraining();
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
        this.trainingController.stop();
    }

    pause() {
        this.trainingController.pause();
    }

    resume() {
        this.trainingController.resume();
    }

    step() {
        this.trainingController.step();
    }

    private async executeTraining(): Promise<void> {
        const { model, datasetManager, callbacks } = this;

        await this.trainModel(model, datasetManager);

        callbacks.onFinished();
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
        const { trainingEventEmitter, callbacks } = this;

        trainingEventEmitter.on('info', callbacks.onInfo);
        trainingEventEmitter.on('error', (message) => {
            callbacks.onError(message);
            this.stop();
        });

        trainingEventEmitter.on('state', (state) => {
            callbacks.onState(state);
            this.handleStateChange(state, this.datasetManager);
        });

        trainingEventEmitter.on('callback', async (params) => {
            await this.handleTrainingIteration(params);
        });
    }

    private async handleTrainingIteration(params: CallbackParameters): Promise<void> {
        const { liveMetrics, callbacks } = this;

        liveMetrics.updateIteration(params);

        const report = await liveMetrics.calculateMetrics();

        callbacks.onReport(report);

        // Handle step-by-step learning mode
        if (this.shouldStopAfterIteration(params.iteration)) {
            this.pause();
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
        this.trainingEventEmitter.clear();
        this.datasetManager.dispose();
        this.liveMetrics.dispose?.();
    }

    private handleTrainingError(error: Error): void {
        console.error('Training failed:', error);
        this.callbacks.onError(`Training failed: ${error.message}`);
    }

    private createModel(
        settings: TrainingSettings,
    ): [PreprocessingModelDecorator<ModelRepresentation>, TrainingEventEmitter, TrainingControl] {
        try {
            const worker = workerRegistry.get(settings.modelSettings.type);
            const eventEmitter = new EventEmitter();
            const trainingController = new TrainingController(eventEmitter);
            const model = worker.modelFactory(settings, eventEmitter, trainingController);
            const pipeline = createPreprocessingPipeline(
                model,
                settings.dataSettings,
                eventEmitter,
            );

            return [pipeline, eventEmitter, trainingController];
        } catch (error) {
            throw new Error(
                `Failed to create model of type ${settings.modelSettings.type}: ${error instanceof Error ? error.message : 'Unknown error'}`,
            );
        }
    }

    private createLiveMetrics(
        settings: TrainingSettings,
        model: Model<ModelRepresentation>,
        datasetManager: DatasetManager,
    ) {
        const worker = workerRegistry.get(settings.modelSettings.type);

        return worker.liveMetricsFactory(model, datasetManager, settings.taskType);
    }
}
