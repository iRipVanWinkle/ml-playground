import { cn } from '@/app/shared/ui/utils';
import type { PredictionComponentProps } from '@/app/shared/registry/types';

export function ClassificationPrediction({
    prediction,
    probabilities,
    categories,
}: PredictionComponentProps) {
    const displayProbabilities =
        probabilities?.length === 1 ? [1 - probabilities[0], probabilities[0]] : probabilities;

    const predictedClass =
        prediction === undefined ? undefined : (categories?.[prediction] ?? String(prediction));

    const sortedProbs = displayProbabilities
        ? displayProbabilities
              .map((value, index) => ({
                  label: categories?.[index] ?? String(index),
                  value,
              }))
              .sort((a, b) => b.value - a.value)
        : [];

    const hasMany = sortedProbs.length > 5;
    const confidence = displayProbabilities?.[prediction ?? -1];

    return (
        <div className="rounded-lg bg-muted/30 p-4 flex flex-col gap-3">
            <div className="text-center text-xs font-medium uppercase tracking-wider text-muted-foreground">
                Prediction
            </div>
            <div className="flex flex-col items-center gap-4">
                <div className="text-center">
                    <div className="text-2xl font-bold text-foreground">
                        {predictedClass ?? '--'}
                    </div>
                    {confidence !== undefined && (
                        <div className="text-sm text-muted-foreground">
                            {(confidence * 100).toFixed(1)}% confidence
                        </div>
                    )}
                </div>

                {sortedProbs.length > 0 &&
                    (hasMany ? (
                        <div className="w-full pt-2">
                            <div className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-3 md:grid-cols-4">
                                {sortedProbs.map(({ label, value }) => {
                                    const isTop = label === predictedClass;
                                    return (
                                        <div
                                            key={label}
                                            className={cn(
                                                'flex items-center justify-between gap-2 rounded px-2 py-1',
                                                isTop && 'bg-primary/10',
                                            )}
                                        >
                                            <span
                                                className={cn(
                                                    'truncate text-sm',
                                                    isTop
                                                        ? 'font-semibold text-foreground'
                                                        : 'text-muted-foreground',
                                                )}
                                            >
                                                {label}
                                            </span>
                                            <span
                                                className={cn(
                                                    'text-sm tabular-nums',
                                                    isTop
                                                        ? 'font-semibold text-foreground'
                                                        : 'text-muted-foreground',
                                                )}
                                            >
                                                {(value * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    ) : (
                        <div className="w-full space-y-2 pt-2">
                            {sortedProbs.map(({ label, value }) => {
                                const isTop = label === predictedClass;
                                return (
                                    <div key={label} className="flex items-center gap-3">
                                        <span
                                            className={cn(
                                                'w-24 truncate text-sm',
                                                isTop
                                                    ? 'font-medium text-foreground'
                                                    : 'text-muted-foreground',
                                            )}
                                        >
                                            {label}
                                        </span>
                                        <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-muted">
                                            <div
                                                className={cn(
                                                    'absolute inset-y-0 left-0 rounded-full transition-all',
                                                    isTop ? 'bg-primary' : 'bg-muted-foreground/30',
                                                )}
                                                style={{ width: `${value * 100}%` }}
                                            />
                                        </div>
                                        <span
                                            className={cn(
                                                'w-12 text-right text-sm tabular-nums',
                                                isTop
                                                    ? 'font-semibold text-foreground'
                                                    : 'text-muted-foreground',
                                            )}
                                        >
                                            {(value * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    ))}
            </div>
        </div>
    );
}
