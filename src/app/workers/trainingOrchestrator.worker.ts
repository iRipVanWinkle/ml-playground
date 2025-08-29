import type { State } from '@/app/store';
import { TrainingOrchestrator } from './training/training-orchestrator';
import type { TrainingState } from '@/ml/types';

interface WorkerMessage {
    type: string;
    payload: string | object;
}

function send(type: string, payload?: string | object, transfer?: Transferable[]) {
    if (transfer) {
        self.postMessage({ type, payload }, { transfer });
    } else {
        self.postMessage({ type, payload });
    }
}

const callbacks = {
    onReport: (report: Float32Array) => send('report', report.buffer, [report.buffer]),
    onState: (state: TrainingState) => send('state', state),
    onInfo: (message: string) => send('info', message),
    onError: (message: string) => send('error', message),
    onFinished: () => send('finished'),
};

let orchestrator: TrainingOrchestrator | null = null;

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
    try {
        const { type, payload } = event.data;

        switch (type) {
            case 'train':
            case 'train-step':
                (async () => {
                    const state = payload as State;
                    orchestrator = await TrainingOrchestrator.createOrchestrator(state, callbacks);
                    orchestrator.train(type === 'train-step');
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
            case 'step':
                orchestrator?.step();
                break;
            default:
                console.warn(`Unknown message type: ${type}`);
        }
    } catch (error) {
        console.error('Worker error:', error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        send('error', `Worker error: ${errorMessage}`);
    }
};
