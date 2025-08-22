import type { State } from '@/app/store';
import { Trainer } from './trainer';
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

const trainer = new Trainer();

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
    const { type, payload } = event.data;

    switch (type) {
        case 'train':
        case 'train-step':
            trainer.train(payload as State, type === 'train-step', {
                onReport: (report: Float32Array) => send('report', report.buffer, [report.buffer]),
                onState: (state: TrainingState) => send('state', state),
                onInfo: (msg: string) => send('info', msg),
                onError: (msg: string) => send('error', msg),
                onFinished: () => send('finished'),
            });
            break;
        case 'stop':
            trainer.stop();
            break;
        case 'pause':
            trainer.pause();
            break;
        case 'resume':
            trainer.resume();
            break;
        case 'step':
            trainer.step();
            break;
        default:
            console.warn(`Unknown message type: ${type}`);
    }
};
