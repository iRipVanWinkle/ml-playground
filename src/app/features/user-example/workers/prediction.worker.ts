import { ready, setBackend, Tensor, tensor2d } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import type { TrainingReport, TrainingSettings } from '@/app/models/types';
import { getWorkerRegistry } from '@/app/models/worker-registry';
import { Randomizer } from '@/ml/random/Randomizer';
import { createPreprocessingPipeline } from '../../control-training/helpers';
import type { UIToWorkerMessage } from './types';

type PredictionPayload = TrainingSettings & {
    example: number[];
    report: TrainingReport;
};

const workerRegistry = getWorkerRegistry();

let backendReady = false;

async function ensureBackend(systemSettings: PredictionPayload['systemSettings']) {
    if (backendReady) return;

    if (systemSettings.backend !== 'auto') {
        const wasmPath =
            import.meta.env.PROD && import.meta.env.BASE_URL !== '/'
                ? `${import.meta.env.BASE_URL}wasm/`
                : '/wasm/';

        setWasmPaths(wasmPath);
        setBackend(systemSettings.backend);
    }

    await ready();

    Randomizer.setSeed(systemSettings.randomSeed);

    backendReady = true;
}

self.onmessage = async (event: MessageEvent<UIToWorkerMessage>) => {
    try {
        const { type, payload, requestId, sentAt } = event.data;

        if (import.meta.env.DEV && sentAt) {
            const now = performance.now() + performance.timeOrigin;
            const latency = now - sentAt;
            console.log(
                `%c[Client -> Worker] %c${type} %clatency: ${latency.toFixed(2)}ms`,
                'color: #ff9800; font-weight: bold',
                'color: inherit',
                'color: #4caf50',
            );
        }

        switch (type) {
            case 'predict': {
                const predictionPayload = payload as PredictionPayload;
                const { modelSettings, dataSettings, systemSettings, example, report } =
                    predictionPayload;

                await ensureBackend(systemSettings);

                const worker = workerRegistry.get(modelSettings.type);

                const parameter = worker.extractParameters(report);
                if (parameter === null) {
                    if (import.meta.env.DEV) {
                        console.warn('No training report provided to prediction worker.');
                    }
                    send('predictions', requestId);
                    return;
                }

                const model = worker.modelFactory(predictionPayload);
                const pipeline = createPreprocessingPipeline(model, dataSettings);

                pipeline.restoreParameters(report.scaler);

                const inputTensor = tensor2d([example]);

                const metadata = pipeline.predictWithMetadata(inputTensor, parameter);

                if (metadata.type === 'regression') {
                    send('predictions', requestId, {
                        type: 'regression',
                        prediction: metadata.predictions.dataSync()[0],
                    });
                }

                if (metadata.type === 'classification') {
                    send('predictions', requestId, {
                        type: 'classification',
                        prediction: metadata.predictions.dataSync()[0],
                        probabilities: metadata.probabilities.dataSync(),
                    });
                }

                if (metadata.type === 'clustering') {
                    send('predictions', requestId, {
                        type: 'clustering',
                        prediction: metadata.assignments.dataSync()[0],
                    });
                }

                if (metadata.type === 'anomaly-detection') {
                    send('predictions', requestId, {
                        type: 'anomaly-detection',
                        prediction: metadata.predictions.dataSync()[0],
                        probabilities: metadata.probabilities.dataSync(),
                    });
                }

                metadata.dispose();
                inputTensor.dispose();
                if (parameter instanceof Tensor) {
                    parameter.dispose();
                }

                model.dispose();

                break;
            }
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        console.error('Worker error:', error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        send('error', undefined, `Worker error: ${errorMessage}`);
    }
};

function send(
    type: string,
    requestId?: string,
    payload?: string | object | Error,
    transfer?: Transferable[],
) {
    const sentAt = import.meta.env.DEV ? performance.now() + performance.timeOrigin : undefined;
    if (transfer) {
        self.postMessage({ type, payload, requestId, sentAt }, { transfer });
    } else {
        self.postMessage({ type, payload, requestId, sentAt });
    }
}
