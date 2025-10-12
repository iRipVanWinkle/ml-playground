import { useEffect, useState } from 'react';
import BackendDetectionWorker from './backend-detection.worker.ts?worker';

type BackendInfo = {
    supported: string[];
    current?: string;
};

export const useBackendDetection = () => {
    const [backendInfo, setBackendInfo] = useState<BackendInfo>({ supported: [] });

    useEffect(() => {
        const worker = new BackendDetectionWorker();

        worker.addEventListener('message', (event: MessageEvent) => {
            if (event.data) {
                try {
                    setBackendInfo(event.data);
                } finally {
                    worker.terminate();
                }
            }
        });

        return () => {
            worker.terminate();
        };
    }, []);

    return backendInfo;
};
