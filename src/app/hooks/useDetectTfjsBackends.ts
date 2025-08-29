import { useEffect, useState } from 'react';
import DetectTfjsBackends from '../workers/detectTfjsBackends.worker.ts?worker';

export const useDetectTfjsBackends = () => {
    const [backendInfo, setBackendInfo] = useState<{ supported: string[], current?: string }>({ supported: [] });

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
