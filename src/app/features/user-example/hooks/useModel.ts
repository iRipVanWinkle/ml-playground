import { useEffect, useRef } from 'react';
import { toast } from 'sonner';
import type { PredictionWorkerManager, UIToWorkerMessage } from '../workers/types';
import { WorkerManager } from '@/app/shared/workers/manager';
import type { TrainingReport } from '@/app/models/types';
import { setUserExamplePrediction, snapshotTrainingSettings } from '@/app/store';

import PredictionWorker from '../workers/prediction.worker.ts?worker';

type UseModelProps = {
    datasetId: string;
};

type PendingPrediction = {
    example: number[];
    report: TrainingReport;
};

export const useModel = ({ datasetId }: UseModelProps) => {
    const pendingRef = useRef<PendingPrediction | null>(null);
    const inFlightRef = useRef(false);
    const dispatchRef = useRef<() => void>(() => {});

    useEffect(() => {
        const wm: PredictionWorkerManager = new WorkerManager<UIToWorkerMessage, TrainingReport>(
            () => new PredictionWorker(),
        );

        const dispatch = () => {
            const pending = pendingRef.current;
            if (!pending || inFlightRef.current) return;

            pendingRef.current = null;
            inFlightRef.current = true;

            wm.postMessage({
                type: 'predict',
                payload: {
                    ...snapshotTrainingSettings(),
                    example: pending.example,
                    report: pending.report,
                },
            });
        };

        dispatchRef.current = dispatch;

        wm.on('predictions', (metadata) => {
            if (metadata) {
                const { prediction, probabilities } = metadata;
                setUserExamplePrediction({
                    prediction,
                    probabilities: probabilities ? Array.from(probabilities) : undefined,
                });
            }
            inFlightRef.current = false;
            dispatch();
        });

        wm.on('error', (error: Error) => {
            console.error(error);
            toast.error(error.message);
            inFlightRef.current = false;
        });

        wm.on('info', (info: string) => {
            console.info(info);
            toast.info(info);
        });

        // Dispatch any request queued before the worker finished initializing.
        dispatch();

        return () => {
            wm.terminate();
            dispatchRef.current = () => {};
            inFlightRef.current = false;
            pendingRef.current = null;
        };
    }, [datasetId]);

    const runPrediction = (example: number[], report: TrainingReport) => {
        pendingRef.current = { example, report };
        dispatchRef.current();
    };

    return { runPrediction };
};
