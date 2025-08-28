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
    onInfo: (msg: string) => send('info', msg),
    onError: (msg: string) => send('error', msg),
    onFinished: () => send('finished'),
};

let trainer: TrainingOrchestrator | null = null;

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
    try {
        const { type, payload } = event.data;

        switch (type) {
            case 'train':
            case 'train-step':
                (async () => {
                    trainer = await TrainingOrchestrator.createOrchestrator(
                        payload as State,
                        callbacks,
                    );
                    trainer.train(type === 'train-step');
                })();
                break;
            case 'stop':
                trainer?.stop();
                break;
            case 'pause':
                trainer?.pause();
                break;
            case 'resume':
                trainer?.resume();
                break;
            case 'step':
                trainer?.step();
                break;
            default:
                console.warn(`Unknown message type: ${type}`);
        }
    } catch (error) {
        console.error('Worker error:', error);
        const message = error instanceof Error ? error.message : String(error);
        send('error', `Worker error: ${message}`);
    }
};
