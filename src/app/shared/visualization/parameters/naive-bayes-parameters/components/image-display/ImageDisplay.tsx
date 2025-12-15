import { row, type TypedArray } from '@/app/shared/helpers';
import {
    CategoryBlock,
    FeatureBlock,
    FeatureHighlight,
    ImageGrid,
} from '@/app/shared/visualization/base';
import type { NaiveBayesParams } from '@/ml/types';
import { createContext, useContext } from 'react';

const ImageGridContext = createContext<{ values: TypedArray }>({
    values: new Float32Array(),
});

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

    return (
        <CategoryBlock key={classIndex} title={categoryName || `Class ${classIndex}`}>
            <FeatureHighlight label="Prior">{prior}</FeatureHighlight>

            <div className="grid grid-cols-2 gap-3">
                <FeatureBlock title="Means">
                    <ImageGridContext.Provider value={{ values: means }}>
                        <ImageGrid
                            values={means}
                            gridSize={calculatedGridSize}
                            tooltipContent={TooltipContent}
                        />
                    </ImageGridContext.Provider>
                </FeatureBlock>

                {!!variances && (
                    <FeatureBlock title="Variances">
                        <ImageGridContext.Provider value={{ values: variances }}>
                            <ImageGrid
                                values={variances}
                                gridSize={calculatedGridSize}
                                tooltipContent={TooltipContent}
                            />
                        </ImageGridContext.Provider>
                    </FeatureBlock>
                )}

                {!!covariances && (
                    <FeatureBlock title="Covariances">
                        <ImageGridContext.Provider value={{ values: covariances.array }}>
                            <ImageGrid
                                values={covariances.array}
                                gridSize={calculatedGridSize * calculatedGridSize}
                                tooltipContent={TooltipContent}
                            />
                        </ImageGridContext.Provider>
                    </FeatureBlock>
                )}
            </div>
        </CategoryBlock>
    );
}

type TooltipContentProps = {
    idx: number;
    gridSize: number;
};

function TooltipContent({ idx, gridSize }: TooltipContentProps) {
    const { values } = useContext(ImageGridContext);
    const value = values?.[idx] ?? 0;
    const row = Math.floor(idx / gridSize);
    const col = idx % gridSize;

    return (
        <p>
            [{row}, {col}]: {value.toFixed(4)}
        </p>
    );
}
