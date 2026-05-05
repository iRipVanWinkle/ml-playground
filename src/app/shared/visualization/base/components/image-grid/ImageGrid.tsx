import { memo, useState, type ComponentType } from 'react';
import type { TypedArray } from '@/app/shared/helpers';
import { Tooltip } from '@/app/shared/ui';
import { useImageHeatmap } from './useImageHeatmap';
import { GRID_TOOLTIP_DELAY_DURATION } from '../../constants';

type TooltipContentProps = {
    idx: number;
    gridSize: number;
};

type ImageGridProps = {
    values: TypedArray;
    gridSize: number;
    tooltipContent: ComponentType<TooltipContentProps>;
};

export function ImageGrid({ values, gridSize, tooltipContent }: ImageGridProps) {
    let min = Infinity;
    let max = -Infinity;
    for (const val of values) {
        min = Math.min(min, val);
        max = Math.max(max, val);
    }

    const imageDataUrl = useImageHeatmap({ values, gridSize, min, max });

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

            <HoverGrid gridSize={gridSize} tooltipContent={tooltipContent} />
        </div>
    );
}

type HoverGridProps = {
    gridSize: number;
    tooltipContent: ComponentType<TooltipContentProps>;
};

const HoverGrid = memo(function HoverGrid({ gridSize, tooltipContent }: HoverGridProps) {
    const [visible, setVisible] = useState(false);
    const cellCount = gridSize * gridSize;

    const TooltipContent = tooltipContent;

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
                            <div className="w-full h-full hover:ring-1 hover:ring-primary/50 hover:z-10 cursor-pointer transition-all duration-200" />
                        </Tooltip.Trigger>
                        <Tooltip.Content>
                            <TooltipContent idx={idx} gridSize={gridSize} />
                        </Tooltip.Content>
                    </Tooltip>
                ))}
        </div>
    );
});
