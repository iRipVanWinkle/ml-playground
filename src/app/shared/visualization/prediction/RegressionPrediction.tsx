import type { PredictionComponentProps } from '@/app/shared/registry/types';

export function RegressionPrediction({ prediction, target }: PredictionComponentProps) {
    return (
        <div className="rounded-lg bg-muted/30 p-4 flex flex-col gap-3">
            <div className="text-center text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Prediction
            </div>
            <div className="flex flex-col items-center gap-1">
                <div className="text-2xl font-bold text-foreground">{prediction.toFixed(4)}</div>
                <div className="text-sm text-muted-foreground">{target}</div>
            </div>
        </div>
    );
}
