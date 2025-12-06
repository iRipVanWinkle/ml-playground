import type { RocCurveData } from '../types';
import { BinaryAUCDisplay } from './BinaryAUCDisplay';
import { MultiAUCDisplay } from './MultiAUCDisplay';

type RocCurveMetricsProps = {
    rocCurveData: RocCurveData;
    categories: string[];
};

export function RocCurveMetrics({ rocCurveData, categories }: RocCurveMetricsProps) {
    return (
        <div className="p-4 rounded-lg bg-muted flex flex-col gap-3">
            {rocCurveData.type === 'binary' ? (
                <BinaryAUCDisplay rocCurveData={rocCurveData} />
            ) : (
                <MultiAUCDisplay rocCurveData={rocCurveData} categories={categories} />
            )}
        </div>
    );
}
