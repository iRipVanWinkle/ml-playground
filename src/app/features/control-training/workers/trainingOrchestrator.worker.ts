import { TrainingOrchestrator } from './training/training-orchestrator';
import type { TrainingState } from '@/ml/types';
import type { UIToWorkerMessage } from './types';
import type { TrainingReport } from '@/app/models/types';

let orchestrator: TrainingOrchestrator | null = null;

self.onmessage = (event: MessageEvent<UIToWorkerMessage>) => {
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
            case 'train':
            case 'train-by-step':
                (async () => {
                    const callbacks = createCallbacks(requestId);

                    orchestrator = await TrainingOrchestrator.createOrchestrator(
                        payload,
                        callbacks,
                    );
                    orchestrator.train(type === 'train-by-step');
                })();
                break;
            case 'stop':
                orchestrator?.stop();
                break;
            case 'pause':
                orchestrator?.pause();
                break;
            case 'resume':
                orchestrator?.resume();
                break;
            case 'step-forward':
                orchestrator?.step();
                break;
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        console.error('Worker error:', error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        send('error', `Worker error: ${errorMessage}`);
    }
};

function createCallbacks(requestId?: string) {
    return {
        onReport: (report: TrainingReport) => send('report', requestId, report),
        onState: (state: TrainingState) => send('state', requestId, state),
        onInfo: (message: string) => send('info', requestId, message),
        onError: (message: string) => send('error', requestId, new Error(message)),
        onFinished: () => send('finished', requestId),
    };
}

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
