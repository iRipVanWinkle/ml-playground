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
} from '@/ml/types';
import type { PreprocessingModelDecorator } from '@/ml/models';
import type { TaskType } from '@/app/shared/types';
import { DatasetManager } from './dataset-manager';
import { TrainingSession } from './training-session';
import { LiveMetricsProps } from './live-metrics-props';
import { LiveMetrics } from './live-metrics';
import { TrainingReportGenerator } from './training-report-generator';
import { accuracy } from '@/ml/metrics/accuracy';
import { createModel, setRandomSeed } from '../../helpers';
import type { SystemSettings } from '@/app/features/system-settings';
import type { TransformationSettings } from '@/app/features/transform-data';
import type { DataState } from '@/app/features/load-dataset/store/types';
import type { ModelSettings } from '@/app/features/configure-model';

type Settings = {
    taskType: TaskType;
    systemSettings: SystemSettings;
    dataSettings: TransformationSettings;
    data: DataState;
    modelSettings: ModelSettings;
};

type TrainingCallbacks = {
    onReport: (report: Float32Array) => void;
    onState: (state: TrainingState) => void;
    onInfo: (msg: string) => void;
    onError: (msg: string) => void;
    onFinished: () => void;
};

type TrainedModel = PreprocessingModelDecorator<ModelRepresentation>;

export class TrainingOrchestrator {
    private model: TrainedModel;
    private datasetManager: DatasetManager;
    private eventEmitter: TrainingEventEmitter;
    private trainingController: TrainingControl;
    private callbacks: TrainingCallbacks;
    private liveMetrics: LiveMetrics;
    private liveMetricsProps: LiveMetricsProps;
    private reportGenerator: TrainingReportGenerator;

    private trainingSession: TrainingSession | null = null;
    private isTraining = false;
    private byStep = false;
    private isClassificationTask = false;

    constructor(settings: Settings, callbacks: TrainingCallbacks) {
        const { modelSettings, dataSettings, systemSettings, data, taskType } = settings;

        const datasetManager = new DatasetManager(data);
        const numFeatures = datasetManager.getTrainingData().X.shape[1];
        const [model, eventEmitter, trainingController] = createModel(
            modelSettings,
            dataSettings,
            taskType,
            numFeatures,
        );

        this.model = model;
        this.callbacks = callbacks;
        this.eventEmitter = eventEmitter;
        this.trainingController = trainingController;
        this.datasetManager = datasetManager;
        this.liveMetricsProps = new LiveMetricsProps(settings);
        this.liveMetrics = new LiveMetrics(model, datasetManager);
        this.reportGenerator = new TrainingReportGenerator();
        this.isClassificationTask = settings.taskType === 'classification';

        setRandomSeed(systemSettings.randomSeed);
        // Set up event handling
        this.setupEventHandlers();
    }

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
        this.trainingController.stop();
        this.isTraining = false;
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

    private async handleTrainingIteration(params: CallbackParameters): Promise<void> {
        if (!this.trainingSession) return;

        // Update training session state
        this.trainingSession.updateIteration(params);

        const metrics = this.isClassificationTask ? [accuracy] : [];
        const liveResults = await this.liveMetrics.calculate(this.trainingSession, metrics);

        const report = this.reportGenerator.generateReport(liveResults, this.trainingSession);

        this.callbacks.onReport(report);

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
        this.eventEmitter.clear();
        this.datasetManager.dispose();
        this.trainingSession?.dispose();
    }

    private handleTrainingError(error: Error): void {
        console.error('Training failed:', error);
        this.callbacks.onError(`Training failed: ${error.message}`);
    }
}
