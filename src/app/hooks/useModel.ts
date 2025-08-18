import { useEffect, useRef } from 'react';
import { toast } from 'sonner';
import {
    resetTrainingReport,
    setTrainingReport,
    setTrainingStatus,
    useAppState,
    type TrainingReport,
} from '@/app/store';
import { decode } from '@/app/helpers/float32Array';
import TrainingWorker from '../workers/training-worker.ts?worker';

function forType<T>(type: string, callback: (payload: T) => void) {
    return (event: MessageEvent) => {
        if (event.data.type === type) {
            callback(event.data.payload);
        }
    };
}

export const useModel = () => {
    const workerRef = useRef<Worker | null>(null);

    useEffect(() => {
        return () => {
            workerRef.current?.terminate();
        };
    }, []);

    const train = async () => {
        setTrainingStatus('training');
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
            forType('error', (msg) => {
                console.error(msg);
                toast.error(msg as string);
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
                worker.terminate();
                workerRef.current = null;
            }),
        );

        worker.addEventListener('error', () => {
            toast.error('An error occurred during training. Please check the console for details.');
            setTrainingStatus('init');
            worker.terminate();
            workerRef.current = null;
        });

        worker.postMessage({
            type: 'train',
            payload: useAppState.getState(),
        });

        workerRef.current = worker;
    };

    const stop = () => {
        setTrainingStatus('init');
        workerRef.current?.postMessage({
            type: 'stop',
        });
    };

    const pause = () => {
        setTrainingStatus('paused');
        workerRef.current?.postMessage({
            type: 'pause',
        });
    };

    const step = () => {
        workerRef.current?.postMessage({
            type: 'step',
        });
    };

    const resume = () => {
        setTrainingStatus('training');
        workerRef.current?.postMessage({
            type: 'resume',
        });
    };

    return { train, stop, pause, step, resume };
};
