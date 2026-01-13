import { useEffect, useRef } from 'react';
import { toast } from 'sonner';
import { setPendingAction, setTrainingStatus } from '../store/actions';
import type { TrainingWorkerManager, UIToWorkerMessage } from '../workers/types';
import { WorkerManager } from '@/app/shared/workers/manager';
import type { TrainingReport, TrainingSettings } from '@/app/models/types';
import { setTrainingReport } from '@/app/features/visualize-training/store/actions';

import TrainingWorker from '../workers/trainingOrchestrator.worker.ts?worker';

type UseModelProps = {
    snapshotTrainingSettings: () => TrainingSettings;
};

export const useModel = ({ snapshotTrainingSettings }: UseModelProps) => {
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

        workerManager.on('report', (report: TrainingReport) => {
            latest = report;
            if (!animationFrame) {
                animationFrame = requestAnimationFrame(() => {
                    setTrainingReport(latest!);
                    animationFrame = null;
                });
            }
        });

        workerManager.on('state', (state: string) => {
            setPendingAction(null);
            switch (state) {
                case 'transforming':
                    setTrainingStatus('preparing');
                    break;
                case 'training':
                    setTrainingStatus('training');
                    break;
                case 'stopped':
                    setTrainingStatus('init');
                    break;
                case 'stepped-forward':
                case 'paused':
                    setTrainingStatus('paused');
                    break;
            }
        });

        workerManager.on('error', (error: Error) => {
            setTrainingStatus('init');
            console.error(error);
            toast.error(error.message);
            terminateWorker();
        });

        workerManager.on('info', (info: string) => {
            console.info(info);
            toast.info(info);
        });

        workerManager.on('finished', () => {
            setTrainingStatus('init');
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

    const stop = () => {
        setPendingAction('stop');
        return workerRef.current?.postMessageAsync({
            type: 'stop',
        });
    };

    const pause = () => {
        setPendingAction('pause');
        return workerRef.current?.postMessageAsync({
            type: 'pause',
        });
    };

    const step = () => {
        setPendingAction('step');
        return workerRef.current?.postMessageAsync({
            type: 'step-forward',
        });
    };

    const resume = () => {
        setPendingAction('resume');
        return workerRef.current?.postMessageAsync({
            type: 'resume',
        });
    };

    return { train, stop, pause, step, resume };
};
