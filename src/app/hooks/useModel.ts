import { useEffect, useRef } from 'react';
import { toast } from 'sonner';
import {
    resetTrainingReport,
    setPendingAction,
    setTrainingReport,
    setTrainingStatus,
    useAppState,
    type TrainingReport,
} from '@/app/store';
import { decode } from '@/app/helpers/float32Array';
import TrainingWorker from '../workers/trainingOrchestrator.worker.ts?worker';

function forType<T>(type: string, callback: (payload: T) => void) {
    return (event: MessageEvent) => {
        if (event.data.type === type) {
            callback(event.data.payload);
        }
    };
}

export const useModel = () => {
    const workerRef = useRef<Worker | null>(null);

    const terminateWorker = () => {
        workerRef.current?.terminate();
        workerRef.current = null;
    };

    useEffect(() => {
        return () => terminateWorker();
    }, []);

    const train = async () => {
        resetTrainingReport();

        if (workerRef.current) return;

        const worker = new TrainingWorker();

        let latest: ArrayBufferLike | null = null;
        let animationFrame: number | null = null;

        worker.addEventListener(
            'message',
            forType('report', (report: ArrayBufferLike) => {
                latest = report;
                if (!animationFrame) {
                    animationFrame = requestAnimationFrame(() => {
                        setTrainingReport(decode<TrainingReport>(new Float32Array(latest!)));
                        animationFrame = null;
                    });
                }
            }),
        );

        worker.addEventListener(
            'message',
            forType('state', (state) => {
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
            }),
        );

        worker.addEventListener(
            'message',
            forType('error', (msg) => {
                setTrainingStatus('init');
                console.error(msg);
                toast.error(msg as string);
                terminateWorker();
            }),
        );

        worker.addEventListener(
            'message',
            forType('info', (msg) => {
                console.info(msg);
                toast.info(msg as string);
            }),
        );

        worker.addEventListener(
            'message',
            forType('finished', () => {
                setTrainingStatus('init');
                terminateWorker();
            }),
        );

        worker.addEventListener('error', (e) => {
            setTrainingStatus('init');
            console.error(e);
            toast.error(e.message);
            terminateWorker();
        });

        worker.postMessage({
            type: 'train',
            payload: useAppState.getState(),
        });

        workerRef.current = worker;
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
            type: 'step',
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
