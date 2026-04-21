import { useEffect, useRef } from 'react';
import { toast } from 'sonner';
import { setTrainingReport, setTrainingState, snapshotTrainingSettings } from '@/app/store';
import type { TrainingWorkerManager, UIToWorkerMessage } from '../workers/types';
import { WorkerManager } from '@/app/shared/workers/manager';
import type { TrainingReport } from '@/app/models/types';
import { uiLogUtils } from '@/app/shared/helpers';

import TrainingWorker from '../workers/trainingOrchestrator.worker.ts?worker';

function getIteration(report: TrainingReport): number | string {
    if ('iterations' in report) return report.iterations[0];
    if ('iteration' in report) return report.iteration;
    return 'N/A';
}

export const useModel = () => {
    const workerRef = useRef<TrainingWorkerManager | null>(null);

    const terminateWorker = () => {
        workerRef.current?.terminate();
        workerRef.current = null;
    };

    useEffect(() => {
        return () => terminateWorker();
    }, []);

    const train = async ({ byStep = false }: { byStep?: boolean } = {}) => {
        if (workerRef.current) return;

        const workerManager: TrainingWorkerManager = new WorkerManager<
            UIToWorkerMessage,
            TrainingReport
        >(() => new TrainingWorker());

        let metricsReceivedCount = 0;

        workerManager.on('report', (report: TrainingReport) => {
            metricsReceivedCount++;

            const iteration = getIteration(report);
            uiLogUtils.logMetricsReceived(metricsReceivedCount, iteration);

            latest = report;
            if (!animationFrame) {
                animationFrame = requestAnimationFrame(() => {
                    try {
                        setTrainingReport(latest!);
                        animationFrame = null;
                    } finally {
                        workerManager.postMessage({ type: 'ready' });
                    }
                });
            }
        });

        workerManager.on('state', (state: string) => {
            switch (state) {
                case 'transforming':
                    setTrainingState('preparing');
                    break;
                case 'training':
                    setTrainingState('training');
                    break;
                case 'stopped':
                    setTrainingState('init');
                    break;
                case 'stepped-forward':
                case 'paused':
                    setTrainingState('paused');
                    break;
            }
        });

        workerManager.on('error', (error: Error) => {
            setTrainingState('init');
            console.error(error);
            toast.error(error.message);
            terminateWorker();
        });

        workerManager.on('info', (info: string) => {
            console.info(info);
            toast.info(info);
        });

        workerManager.on('finished', () => {
            uiLogUtils.logTrainingComplete(metricsReceivedCount);

            setTrainingState('init');
            terminateWorker();
            toast.success('Training finished');
        });

        let latest: TrainingReport | null = null;
        let animationFrame: number | null = null;

        workerManager.postMessage({
            type: byStep ? 'train-by-step' : 'train',
            payload: snapshotTrainingSettings(),
        });

        workerRef.current = workerManager;
    };

    const stop = () =>
        workerRef.current?.postMessageAsync({
            type: 'stop',
        });

    const pause = () =>
        workerRef.current?.postMessageAsync({
            type: 'pause',
        });

    const step = () =>
        workerRef.current?.postMessageAsync({
            type: 'step-forward',
        });

    const resume = () =>
        workerRef.current?.postMessageAsync({
            type: 'resume',
        });

    return { train, stop, pause, step, resume };
};
