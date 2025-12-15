import { createContext, useContext } from 'react';
import type { MatrixLike, TypedArray } from '@/app/shared/helpers';
import { useClassParameters } from '../hooks';
import { FeatureHighlight, ImageGrid } from '../../../base';

const ImageGridContext = createContext<{ weights: TypedArray }>({
    weights: new Float32Array(0),
});

type ImageParametersProps = {
    theta: MatrixLike;
    categories?: string[];
    selectedClassIndex?: number;
};

export function ImageParameters({ theta, categories, selectedClassIndex }: ImageParametersProps) {
    const [, numFeatures] = theta.shape;
    const numWeights = numFeatures - 1; // exclude bias
    const calculatedGridSize = Math.floor(Math.sqrt(numWeights));

    const classParameters = useClassParameters(theta);

    const displayedClasses = classParameters.filter(
        ({ classIndex }) => selectedClassIndex === undefined || classIndex === selectedClassIndex,
    );

    return (
        <div className="flex flex-col gap-3">
            {displayedClasses.map(({ classIndex, bias, weights }) => {
                const categoryName = categories?.[classIndex] || `Class ${classIndex}`;
                return (
                    <div
                        key={categoryName}
                        className="rounded-lg bg-primary-foreground p-4 flex flex-col gap-3"
                    >
                        <h4 className="text-base font-semibold text-primary">{categoryName}</h4>

                        <FeatureHighlight label="Intercept (Bias)">{bias}</FeatureHighlight>

                        <ImageGridContext.Provider value={{ weights }}>
                            <ImageGrid
                                values={weights}
                                gridSize={calculatedGridSize}
                                tooltipContent={TooltipContent}
                            />
                        </ImageGridContext.Provider>
                    </div>
                );
            })}
        </div>
    );
}

type TooltipContentProps = {
    idx: number;
    gridSize: number;
};

function TooltipContent({ idx, gridSize }: TooltipContentProps) {
    const { weights } = useContext(ImageGridContext);
    const weight = weights?.[idx] ?? 0;
    const row = Math.floor(idx / gridSize);
    const col = idx % gridSize;

    return (
        <p>
            [{row}, {col}]: {weight.toFixed(4)}
        </p>
    );
}
