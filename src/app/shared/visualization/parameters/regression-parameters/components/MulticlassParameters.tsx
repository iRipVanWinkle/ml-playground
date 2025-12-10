import { useMemo } from 'react';
import { ParametersDisplay } from './ParametersDisplay';
import type { MatrixLike } from '@/app/shared/helpers';

type MulticlassParametersProps = {
    theta: MatrixLike;
    featureLabels: string[];
    categories?: string[];
    selectedClassIndex?: number;
};

export function MulticlassParameters({
    theta,
    featureLabels,
    categories,
    selectedClassIndex,
}: MulticlassParametersProps) {
    const [numClasses, numFeatures] = theta.shape;

    const classParameters = useMemo(() => {
        return Array.from({ length: numClasses }, (_, c) => {
            const rowStart = c * numFeatures;

            return {
                classIndex: c,
                bias: theta.array[rowStart],
                weights: Array.from(theta.array.subarray(rowStart + 1, rowStart + numFeatures)),
            };
        });
    }, [theta, numClasses, numFeatures]);

    const displayedClasses = classParameters.filter(
        ({ classIndex }) => selectedClassIndex === undefined || classIndex === selectedClassIndex,
    );

    return (
        <div className="flex flex-col gap-3">
            {displayedClasses.map(({ classIndex, bias, weights }) => {
                const categoryName = categories?.[classIndex] || `Class ${classIndex}`;
                return (
                    <ParametersDisplay
                        key={classIndex}
                        categoryName={categoryName}
                        bias={bias}
                        weights={weights}
                        featureLabels={featureLabels}
                    />
                );
            })}

            <div className="rounded-lg border bg-muted/50 p-4">
                <div className="mb-2 text-sm font-medium text-muted-foreground">Model Equation</div>
                <div className="overflow-x-auto font-mono text-sm">
                    <div>P(y=k) = softmax(z₁, z₂, ..., zₖ)</div>
                    <div className="mt-2 text-muted-foreground">
                        where zₖ = biasₖ + Σ(weightₖᵢ × featureᵢ)
                    </div>
                </div>
            </div>
        </div>
    );
}
