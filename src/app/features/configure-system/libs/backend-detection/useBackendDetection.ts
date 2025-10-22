import { useEffect, useState } from 'react';
import { WorkerManager } from '@/app/shared/workers';
import BackendDetectionWorker from './backend-detection.worker.ts?worker';

type BackendInfo = {
    supported: string[];
    current?: string;
};

type BackendDetectionMessage = {
    type: 'detect-backends';
    requestId?: string;
};

export const useBackendDetection = () => {
    const [backendInfo, setBackendInfo] = useState<BackendInfo>({ supported: [] });
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<Error | null>(null);

    useEffect(() => {
        const workerManager = new WorkerManager<BackendDetectionMessage, BackendInfo>(
            () => new BackendDetectionWorker(),
        );

        const detectBackends = async () => {
            try {
                setIsLoading(true);
                setError(null);

                const result = await workerManager.postMessageAsync(
                    { type: 'detect-backends' },
                    { timeout: 10000 }, // 10 second timeout
                );

                setBackendInfo(result);
            } catch (err) {
                setError(err instanceof Error ? err : new Error('Unknown error occurred'));
                setBackendInfo({ supported: [] });
            } finally {
                setIsLoading(false);
                workerManager.terminate();
            }
        };

        detectBackends();

        return () => {
            workerManager.terminate();
        };
    }, []);

    return { backendInfo, isLoading, error };
};
