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
        </div>
    );
}
