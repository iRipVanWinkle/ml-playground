import { row } from '@/app/shared/helpers';
import { CategoryBlock, FeatureBlock, FeatureHighlight } from '@/app/shared/visualization/base';
import type { NaiveBayesParams } from '@/ml/types';
import { ImageGrid } from './ImageGrid';
import { useMemo } from 'react';

type ImageDisplayProps = {
    params: NaiveBayesParams;
    classIndex: number;
    categoryName?: string;
};

export function ImageDisplay({ params, classIndex, categoryName }: ImageDisplayProps) {
    const prior = params.classPriors[classIndex];
    const means = row(params.classMeans, classIndex);
    const variances =
        params.type === 'gaussian' ? row(params.classVariances, classIndex) : undefined;
    const covariances =
        params.type === 'quadratic' ? params.classCovariances[classIndex] : undefined;

    const numFeatures = means.length;
    const calculatedGridSize = Math.floor(Math.sqrt(numFeatures));

    const { minMean, maxMean, minVar, maxVar, minCov, maxCov } = useMemo(() => {
        let minMean = Infinity;
        let maxMean = -Infinity;
        let minVar = Infinity;
        let maxVar = -Infinity;
        let minCov = Infinity;
        let maxCov = -Infinity;

        for (let i = 0; i < numFeatures; i++) {
            const mean = means[i] ?? 0;
            minMean = Math.min(minMean, mean);
            maxMean = Math.max(maxMean, mean);

            if (variances) {
                const variance = variances[i] ?? 0;
                minVar = Math.min(minVar, variance);
                maxVar = Math.max(maxVar, variance);
            }

            if (covariances) {
                const rowOffset = i * numFeatures;
                for (let j = 0; j < numFeatures; j++) {
                    const covariance = covariances.array[rowOffset + j];
                    minCov = Math.min(minCov, covariance);
                    maxCov = Math.max(maxCov, covariance);
                }
            }
        }

        return { minMean, maxMean, minVar, maxVar, minCov, maxCov };
    }, [means, variances, covariances, numFeatures]);

    return (
        <CategoryBlock key={classIndex} title={categoryName || `Class ${classIndex}`}>
            <FeatureHighlight label="Prior">{prior}</FeatureHighlight>

            <div className="grid grid-cols-2 gap-3">
                <FeatureBlock title="Means">
                    <ImageGrid
                        values={means}
                        gridSize={calculatedGridSize}
                        min={minMean}
                        max={maxMean}
                    />
                </FeatureBlock>

                {!!variances && (
                    <FeatureBlock title="Variances">
                        <ImageGrid
                            values={variances}
                            gridSize={calculatedGridSize}
                            min={minVar}
                            max={maxVar}
                        />
                    </FeatureBlock>
                )}

                {!!covariances && (
                    <FeatureBlock title="Covariances">
                        <ImageGrid
                            values={covariances.array}
                            gridSize={calculatedGridSize * calculatedGridSize}
                            min={minCov}
                            max={maxCov}
                        />
                    </FeatureBlock>
                )}
            </div>
        </CategoryBlock>
    );
}
