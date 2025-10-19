import { useEffect, useRef } from 'react';
import { toast } from 'sonner';
import TrainingWorker from '../workers/trainingOrchestrator.worker.ts?worker';
import { decode } from '../helpers';
import { useSystemStore } from '../../system-settings';
import { useTransformationStore } from '../../transform-data';
import { useDatasetStore } from '../../load-dataset';
import { useModelSettingsStore } from '../../configure-model';
import { reset, setPendingAction, setTrainingReport, setTrainingStatus } from '../store/actions';
import type { TrainingReport } from '../store';
import { useTaskType } from '@/app/features/task-switcher';
import type { TrainingWorkerManager, UIToWorkerMessage } from '../workers/types';
import { WorkerManager } from '@/app/shared/workers';

export const useModel = () => {
    const taskType = useTaskType();
    const workerRef = useRef<TrainingWorkerManager | null>(null);

    const terminateWorker = () => {
        workerRef.current?.terminate();
        workerRef.current = null;
    };

    useEffect(() => {
        return () => terminateWorker();
    }, []);

    const train = async ({ byStep = false }: { byStep?: boolean } = {}) => {
        reset();

        if (workerRef.current) return;

        const workerManager: TrainingWorkerManager = new WorkerManager<
            UIToWorkerMessage,
            TrainingReport
        >(() => new TrainingWorker());

        workerManager.on('report', (report: ArrayBufferLike) => {
            latest = report;
            if (!animationFrame) {
                animationFrame = requestAnimationFrame(() => {
                    setTrainingReport(decode<TrainingReport>(new Float32Array(latest!)));
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

        let latest: ArrayBufferLike | null = null;
        let animationFrame: number | null = null;

        workerManager.postMessage({
            type: byStep ? 'train-by-step' : 'train',
            payload: {
                taskType,
                systemSettings: useSystemStore.getState(),
                dataSettings: useTransformationStore.getState(),
                data: useDatasetStore.getState(),
                modelSettings: useModelSettingsStore.getState(),
            },
        });

        workerRef.current = workerManager;
    };

    const stop = () => {
        setPendingAction('stop');
        workerRef.current?.postMessage({
            type: 'stop',
        });
    };

    const pause = () => {
        setPendingAction('pause');
        workerRef.current?.postMessage({
            type: 'pause',
        });
    };

    const step = () => {
        setPendingAction('step');
        workerRef.current?.postMessage({
            type: 'step-forward',
        });
    };

    const resume = () => {
        setPendingAction('resume');
        workerRef.current?.postMessage({
            type: 'resume',
        });
    };

    return { train, stop, pause, step, resume };
};
