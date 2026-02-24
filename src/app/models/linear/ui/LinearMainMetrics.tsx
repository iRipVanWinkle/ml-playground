import type { LinearTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

export function LinearMainMetrics({ report }: MainMetricsProps<LinearTrainingReport>) {
    const { trainLoss, testLoss } = report;

    return (
        <>
            <div>
                Train Loss:{' '}
                <div className="font-bold tabular-nums">
                    {trainLoss ? trainLoss.toFixed(4) : '--'}
                </div>
            </div>
            <div>
                Test Loss:{' '}
                <div className="font-bold tabular-nums">
                    {testLoss ? testLoss.toFixed(4) : '--'}
                </div>
            </div>
        </>
    );
}
