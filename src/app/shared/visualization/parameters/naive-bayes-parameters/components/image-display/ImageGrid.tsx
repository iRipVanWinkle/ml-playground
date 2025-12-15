import type { TypedArray } from '@/app/shared/helpers';
import { useImageHeatmap } from '../../hooks';
import { createContext, memo, useContext, useMemo, useState } from 'react';
import { GRID_TOOLTIP_DELAY_DURATION } from '../../constants';
import { Tooltip } from '@/app/shared/ui';

const ImageGridContext = createContext<{ values: TypedArray }>({
    values: new Float32Array(),
});

type ImageGridProps = {
    values: TypedArray;
    gridSize: number;
    min: number;
    max: number;
};

export function ImageGrid({ values, gridSize, min, max }: ImageGridProps) {
    const imageDataUrl = useImageHeatmap({ values, gridSize, min, max });
    const contextValue = useMemo(() => ({ values }), [values]);

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
            onPointerEnter={gridSize <= 100 ? handleVisibilityChange : undefined}
        >
            {visible &&
                Array.from({ length: cellCount }, (_, idx) => (
                    <Tooltip delayDuration={GRID_TOOLTIP_DELAY_DURATION} key={idx}>
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
