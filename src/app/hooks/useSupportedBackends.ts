import { useEffect, useState } from 'react';
import SupportedBackendsWorker from '../workers/supported-backends-worker.ts?worker';

export const useSupportedBackends = () => {
    const [supportedBackends, setSupportedBackends] = useState<string[]>([]);

    useEffect(() => {
        const worker = new SupportedBackendsWorker();

        worker.addEventListener('message', (event: MessageEvent) => {
            if (event.data) {
                setSupportedBackends(event.data);
            }
        });

        return () => {
            worker.terminate();
        };
    }, []);

    return supportedBackends;
};
