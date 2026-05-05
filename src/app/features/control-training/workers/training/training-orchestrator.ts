import { Tensor, ready, setBackend } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import type {
    CallbackParameters,
    TrainingControl,
    TrainingEventEmitter,
    TrainingState,
    ScalerState,
} from '@/ml/types';
import { PipelineModel } from '@/ml/models';
import { EventEmitter } from '@/ml/events/EventEmitter';
import { TrainingController } from '@/ml/controllers/TrainingController';
import { Randomizer } from '@/ml/random/Randomizer';
import {
    DatasetManager,
    type LiveMetrics,
    performanceUtils,
    MemoryLeakDetector,
    workerLogUtils,
} from '@/app/shared/workers';
import type { ModelType, TrainingReport } from '@/app/models/types';
import { getWorkerRegistry } from '@/app/models/worker-registry';
import type { TrainingSettings } from '@/app/models/types';
import type { RepresentationOf } from '@/app/shared/registry';
import { createPreprocessingPipeline } from '../../helpers';

type TrainingCallbacks = {
    onReport: (report: TrainingReport) => void;
    onState: (state: TrainingState) => void;
    onInfo: (msg: string) => void;
    onError: (msg: string) => void;
    onFinished: () => void;
};

type TrainedModel = PipelineModel<RepresentationOf<ModelType>>;

const workerRegistry = getWorkerRegistry();

export class TrainingOrchestrator {
    private model: TrainedModel;
    private datasetManager: DatasetManager;
    private trainingEventEmitter: TrainingEventEmitter;
    private trainingController: TrainingControl;
    private callbacks: TrainingCallbacks;
    private liveMetrics: LiveMetrics<CallbackParameters, TrainingReport>;
    private memoryLeakDetector: MemoryLeakDetector;

    private isTraining = false;
    private byStep = false;
    private isReady = true;
    private pendingReport: TrainingReport | null = null;

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
        this.memoryLeakDetector = new MemoryLeakDetector();

        this.callbacks = callbacks;

        Randomizer.setSeed(systemSettings.randomSeed);
        this.setupEventHandlers();
    }

    async train(byStep: boolean): Promise<void> {
        if (this.isTraining) {
            throw new Error('Training already in progress');
        }

        this.byStep = byStep;
        this.isTraining = true;

        workerLogUtils.logTrainingLifecycle('started');

        try {
            await this.executeTraining();
        } catch (error) {
            const prepareError = error instanceof Error ? error : new Error(String(error));
            this.handleTrainingError(prepareError);
        } finally {
            this.cleanup();
            this.isTraining = false;
        }

        workerLogUtils.logTrainingLifecycle('finished');
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

    setReady(ready: boolean) {
        this.isReady = ready;

        if (ready && this.pendingReport) {
            this.callbacks.onReport(this.pendingReport);
            this.pendingReport = null;
            this.isReady = false;
        }
    }

    private async executeTraining(): Promise<void> {
        const { model, datasetManager, callbacks } = this;

        await this.trainModel(model, datasetManager);

        if (this.pendingReport) {
            callbacks.onReport(this.pendingReport);
            this.pendingReport = null;
        }

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
            // Deep-clone so each report owns its scaler buffers — required because
            // the report's typed-array buffers are transferred to the main thread.
            report.scaler = structuredClone(this.scalerParams);
        }

        if (import.meta.env.DEV) {
            performanceUtils.logDuration('[Worker]', 'calculateMetrics', startTime);
            this.memoryLeakDetector.check();
        }

        if (this.isReady) {
            callbacks.onReport(report);
            this.isReady = false;
        } else {
            this.pendingReport = report;
        }

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
        this.memoryLeakDetector.reset();
        this.pendingReport = null;
        this.isReady = true;
        this.scalerParams = undefined;
        this.hasExtractedScalerParams = false;
    }

    private handleTrainingError(error: Error): void {
        workerLogUtils.logError('Training failed:', error);
        this.callbacks.onError(`Training failed: ${error.message}`);
    }

    private createModel(
        settings: TrainingSettings,
    ): [PipelineModel<RepresentationOf<ModelType>>, TrainingEventEmitter, TrainingControl] {
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
}
