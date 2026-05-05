import { TrainingOrchestrator } from './training/training-orchestrator';
import type { TrainingState } from '@/ml/types';
import type { UIToWorkerMessage } from './types';
import type { TrainingReport } from '@/app/models/types';
import { collectTransferables, performanceUtils, workerLogUtils } from '@/app/shared/workers';

let orchestrator: TrainingOrchestrator | null = null;

self.onmessage = (event: MessageEvent<UIToWorkerMessage>) => {
    try {
        const { type, payload, requestId, sentAt } = event.data;

        performanceUtils.logLatency('[Client -> Worker]', type, sentAt, '#ff9800');

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
                send('state', requestId, 'stopped');
                break;
            case 'pause':
                orchestrator?.pause();
                send('state', requestId, 'paused');
                break;
            case 'resume':
                orchestrator?.resume();
                send('state', requestId, 'training');
                break;
            case 'step-forward':
                orchestrator?.step();
                send('state', requestId, 'stepped-forward');
                break;
            case 'ready':
                orchestrator?.setReady(true);
                break;
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        workerLogUtils.logError('Worker error:', error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        send('error', `Worker error: ${errorMessage}`);
    }
};

function createCallbacks(requestId?: string) {
    return {
        onReport: (report: TrainingReport) =>
            send('report', requestId, report, collectTransferables(report)),
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
    const sentAt = performanceUtils.getTimestamp();
    if (transfer) {
        self.postMessage({ type, payload, requestId, sentAt }, { transfer });
    } else {
        self.postMessage({ type, payload, requestId, sentAt });
    }
}
