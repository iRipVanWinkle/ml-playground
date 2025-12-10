import { memo, createContext, useContext, useState, useMemo } from 'react';
import type { MatrixLike, TypedArray } from '@/app/shared/helpers';
import { BiasTerm } from './BiasTerm';
import { useClassParameters, useImageHeatmap } from '../hooks';
import { Tooltip } from '@/app/shared/ui';

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
            {displayedClasses.map(({ classIndex, bias, weights, min, max }) => {
                const categoryName = categories?.[classIndex] || `Class ${classIndex}`;
                return (
                    <div
                        key={categoryName}
                        className="rounded-lg bg-primary-foreground p-4 flex flex-col gap-3"
                    >
                        <h4 className="text-base font-semibold text-primary">{categoryName}</h4>

                        <BiasTerm bias={bias} />

                        <ImageGrid
                            weights={weights}
                            gridSize={calculatedGridSize}
                            min={min}
                            max={max}
                        />
                    </div>
                );
            })}
        </div>
    );
}

type ImageGridProps = {
    weights: TypedArray;
    gridSize: number;
    min: number;
    max: number;
};

const ImageGridContext = createContext<{ weights: TypedArray }>({
    weights: new Float32Array(0),
});

function ImageGrid({ weights, gridSize, min, max }: ImageGridProps) {
    const imageDataUrl = useImageHeatmap(weights, gridSize, min, max);
    const contextValue = useMemo(() => ({ weights }), [weights]);

    return (
        <div className="relative w-full" style={{ aspectRatio: '1 / 1' }}>
            <div
                className="absolute inset-0"
                style={{
                    backgroundImage: `url(${imageDataUrl})`,
                    backgroundSize: '100% 100%',
                    backgroundRepeat: 'no-repeat',
                    imageRendering: 'pixelated',
                }}
            />

            <ImageGridContext.Provider value={contextValue}>
                <HoverGrid gridSize={gridSize} />
            </ImageGridContext.Provider>
        </div>
    );
}

type HoverGridProps = {
    gridSize: number;
};

const DELAY_DURATION = 250;

const HoverGrid = memo(function HoverGrid({ gridSize }: HoverGridProps) {
    const [visible, setVisible] = useState(false);
    const cellCount = gridSize * gridSize;

    const handleVisibilityChange = () => {
        setVisible(true);
    };

    return (
        <div
            className="absolute inset-0 grid"
            style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}
            onPointerEnter={handleVisibilityChange}
        >
            {visible &&
                Array.from({ length: cellCount }, (_, idx) => (
                    <Tooltip delayDuration={DELAY_DURATION} key={idx}>
                        <Tooltip.Trigger asChild>
                            <div className="hover:ring-1" />
                        </Tooltip.Trigger>
                        <Tooltip.Content>
                            <TooltipContent idx={idx} gridSize={gridSize} />
                        </Tooltip.Content>
                    </Tooltip>
                ))}
        </div>
    );
});

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
