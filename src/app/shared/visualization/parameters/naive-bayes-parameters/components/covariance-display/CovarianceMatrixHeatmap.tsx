import { memo, useMemo, useState } from 'react';
import { useImageHeatmap } from '../../hooks';
import { type MatrixLike } from '@/app/shared/helpers';
import { TooltipWrapper } from './TooltipContent';

type CovarianceMatrixHeatmapProps = {
    covariances: MatrixLike;
};

export function CovarianceMatrixHeatmap({ covariances }: CovarianceMatrixHeatmapProps) {
    const gridSize = covariances.shape[0];
    const { min, max } = useMemo(() => {
        let min = Infinity;
        let max = -Infinity;
        for (const val of covariances.array) {
            if (val < min) min = val;
            if (val > max) max = val;
        }
        return { min, max };
    }, [covariances]);

    const imageDataUrl = useImageHeatmap({
        values: covariances.array,
        gridSize,
        min,
        max,
        showDiagonal: true,
    });

    return (
        <div className="flex flex-col gap-2">
            <div className="relative w-full" style={{ aspectRatio: '1 / 1' }}>
                <div
                    className="absolute inset-0 rounded"
                    style={{
                        backgroundImage: `url(${imageDataUrl})`,
                        backgroundSize: '100% 100%',
                        backgroundRepeat: 'no-repeat',
                        imageRendering: 'pixelated',
                    }}
                />

                <HoverGrid gridSize={gridSize} />
            </div>
        </div>
    );
}

type HoverGridProps = {
    gridSize: number;
};

const HoverGrid = memo(function HoverGrid({ gridSize }: HoverGridProps) {
    const cellCount = gridSize * gridSize;

    const [visible, setVisible] = useState(false);

    const handleVisibilityChange = () => {
        setVisible(true);
    };

    return (
        <div
            className="absolute inset-0 grid rounded"
            style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}
            onPointerEnter={handleVisibilityChange}
        >
            {visible &&
                Array.from({ length: cellCount }, (_, idx) => (
                    <TooltipWrapper
                        rowIndex={Math.floor(idx / gridSize)}
                        colIndex={idx % gridSize}
                        key={idx}
                    >
                        <div className="w-full h-full hover:ring-1 hover:ring-primary/50 hover:z-10 cursor-pointer transition-all duration-200" />
                    </TooltipWrapper>
                ))}
        </div>
    );
});
