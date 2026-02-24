import { Tensor, memory, ready, setBackend } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import type {
    CallbackParameters,
    ModelRepresentation,
    TrainingControl,
    TrainingEventEmitter,
    TrainingState,
    ScalerState,
} from '@/ml/types';
import { PipelineModel } from '@/ml/models';
import { DatasetManager } from '@/app/shared/workers';
import { createPreprocessingPipeline } from '../../helpers';
import type { TrainingReport } from '@/app/models/types';
import { getWorkerRegistry } from '@/app/models/worker-registry';
import type { TrainingSettings } from '@/app/models/types';
import { EventEmitter } from '@/ml/events/EventEmitter';
import { TrainingController } from '@/ml/controllers/TrainingController';
import type { LiveMetrics } from '@/app/shared/workers';
import { Randomizer } from '@/ml/random/Randomizer';

type TrainingCallbacks = {
    onReport: (report: TrainingReport) => void;
    onState: (state: TrainingState) => void;
    onInfo: (msg: string) => void;
    onError: (msg: string) => void;
    onFinished: () => void;
};

type TrainedModel = PipelineModel<ModelRepresentation>;

const workerRegistry = getWorkerRegistry();

const CONSECUTIVE_TENSOR_INCREASE_THRESHOLD = 5;

export class TrainingOrchestrator {
    private model: TrainedModel;
    private datasetManager: DatasetManager;
    private trainingEventEmitter: TrainingEventEmitter;
    private trainingController: TrainingControl;
    private callbacks: TrainingCallbacks;
    private liveMetrics: LiveMetrics<CallbackParameters, TrainingReport>;

    private isTraining = false;
    private byStep = false;
    private previousTensorCount: number | null = null;
    private consecutiveTensorIncreases = 0;

    private scalerParams?: ScalerState;
    private hasExtractedScalerParams = false;

    static async createOrchestrator(
        settings: TrainingSettings,
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
        const { systemSettings, dataset } = settings;

        [this.model, this.trainingEventEmitter, this.trainingController] =
            this.createModel(settings);

        this.datasetManager = new DatasetManager(dataset);
        this.liveMetrics = this.createLiveMetrics(settings, this.model, this.datasetManager);

        this.callbacks = callbacks;

        Randomizer.setSeed(systemSettings.randomSeed);
        // Set up event handling
        this.setupEventHandlers();
        this.initializeTensorTracking();
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

        const startTime = import.meta.env.DEV ? performance.now() : 0;

        if (!this.hasExtractedScalerParams) {
            this.scalerParams = await this.model.extractScalerParams();
            this.hasExtractedScalerParams = true;
        }

        const report = await liveMetrics.calculateMetrics(params);

        if (this.scalerParams) {
            report.scaler = this.scalerParams;
        }

        if (import.meta.env.DEV) {
            const duration = performance.now() - startTime;
            console.log(
                `%c[Worker] %ccalculateMetrics %cduration: ${duration.toFixed(2)}ms`,
                'color: #9c27b0; font-weight: bold',
                'color: inherit',
                'color: #4caf50',
            );

            this.checkTensorMemory();
        }

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
            const trainingData = datasetManager.getTrainingData();
            const testData = datasetManager.getTestData();
            const predictionData = datasetManager.getPredictionData();

            this.model.prepareFeatures(trainingData.X, true);
            this.model.prepareLabels(trainingData.y);

            if (testData) {
                this.model.prepareFeatures(testData.X);
                this.model.prepareLabels(testData.y);
            }

            if (predictionData) {
                this.model.prepareFeatures(predictionData);
            }
        }
    }

    private cleanup(): void {
        this.model.dispose(true);
        this.trainingEventEmitter.clear();
        this.datasetManager.dispose();
        this.liveMetrics.dispose?.();
        this.resetTensorTracking();
        this.scalerParams = undefined;
        this.hasExtractedScalerParams = false;
    }

    private handleTrainingError(error: Error): void {
        console.error('Training failed:', error);
        this.callbacks.onError(`Training failed: ${error.message}`);
    }

    private createModel(
        settings: TrainingSettings,
    ): [PipelineModel<ModelRepresentation>, TrainingEventEmitter, TrainingControl] {
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
        model: TrainedModel,
        datasetManager: DatasetManager,
    ) {
        const worker = workerRegistry.get(settings.modelSettings.type);

        return worker.liveMetricsFactory(model, datasetManager, settings);
    }

    private initializeTensorTracking(): void {
        this.previousTensorCount = memory().numTensors;
        this.consecutiveTensorIncreases = 0;
    }

    private checkTensorMemory(): void {
        const currentMemory = memory();
        const currentTensorCount = currentMemory.numTensors;

        if (this.previousTensorCount !== null) {
            if (currentTensorCount > this.previousTensorCount) {
                this.consecutiveTensorIncreases++;

                if (this.consecutiveTensorIncreases >= CONSECUTIVE_TENSOR_INCREASE_THRESHOLD) {
                    console.warn(
                        `%c[Worker] %cMemory Leak Warning: %cTensor count has increased for ${this.consecutiveTensorIncreases} consecutive iterations (${this.previousTensorCount} → ${currentTensorCount})`,
                        'color: #ff9800; font-weight: bold',
                        'color: #f44336; font-weight: bold',
                        'color: inherit',
                    );
                    console.warn('Current memory:', currentMemory);
                }
            } else {
                this.consecutiveTensorIncreases = 0;
            }
        }

        this.previousTensorCount = currentTensorCount;
    }

    private resetTensorTracking(): void {
        this.previousTensorCount = null;
        this.consecutiveTensorIncreases = 0;
    }
}
