import { cn } from '@/app/shared/ui/utils';
import type { PredictionComponentProps } from '@/app/shared/registry/types';

export function AnomalyPrediction({ prediction, probabilities }: PredictionComponentProps) {
    const isAnomaly = prediction === -1;
    const score = probabilities?.[0];

    return (
        <div className="rounded-lg bg-muted/30 p-4 flex flex-col gap-3">
            <div className="text-center text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Prediction
            </div>
            <div className="flex flex-col items-center gap-1">
                <div
                    className={cn(
                        'text-2xl font-bold',
                        isAnomaly ? 'text-destructive' : 'text-foreground',
                    )}
                >
                    {isAnomaly ? 'Anomaly' : 'Normal'}
                </div>
                {score !== undefined && (
                    <div className="text-sm text-muted-foreground">Score: {score.toFixed(3)}</div>
                )}
            </div>
        </div>
    );
}
