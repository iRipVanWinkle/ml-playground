import { useEffect, useState } from 'react';
import DetectTfjsBackends from './detectTfjsBackends.worker.ts?worker';

type BackendInfo = {
    supported: string[];
    current?: string;
};

export const useDetectTfjsBackends = () => {
    const [backendInfo, setBackendInfo] = useState<BackendInfo>({ supported: [] });

    useEffect(() => {
        const worker = new DetectTfjsBackends();

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
