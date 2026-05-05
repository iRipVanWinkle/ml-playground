import { Fragment, type ComponentType } from 'react';
import { type MatrixLike } from '@/app/shared/helpers';
import { Tooltip } from '@/app/shared/ui';
import { MatrixGrid, GRID_TOOLTIP_DELAY_DURATION } from '@/app/shared/visualization/base';

type TooltipContentProps = {
    idx: number;
    gridSize: number;
};

type CovarianceMatrixGridProps = {
    covariances: MatrixLike;
    featureLabels: string[];
    tooltipContent: ComponentType<TooltipContentProps>;
};

export function CovarianceMatrixGrid({
    covariances,
    featureLabels,
    tooltipContent,
}: CovarianceMatrixGridProps) {
    const size = covariances.shape[0];
    let absMax = -Infinity;
    for (const val of covariances.array) {
        const absVal = Math.abs(val);
        if (absVal > absMax) absMax = absVal;
    }

    return (
        <MatrixGrid size={size}>
            {featureLabels.map((label, idx) => (
                <MatrixGrid.ColTitle key={`col-${idx}`}>{label}</MatrixGrid.ColTitle>
            ))}

            {Array.from({ length: size }, (_, rowIndex) => (
                <Fragment key={`row-${rowIndex}`}>
                    <MatrixGrid.RowTitle>{featureLabels[rowIndex]}</MatrixGrid.RowTitle>

                    {Array.from({ length: size }, (_, colIndex) => {
                        const value = covariances.array[rowIndex * size + colIndex];
                        return (
                            <CovarianceCell
                                key={`cell-${rowIndex}-${colIndex}`}
                                value={value}
                                absMax={absMax}
                                idx={rowIndex * size + colIndex}
                                gridSize={size}
                                tooltipContent={tooltipContent}
                            />
                        );
                    })}
                </Fragment>
            ))}
        </MatrixGrid>
    );
}

type CovarianceCellProps = {
    value: number;
    absMax: number;
    idx: number;
    gridSize: number;
    tooltipContent: ComponentType<TooltipContentProps>;
};

function CovarianceCell({ value, absMax, idx, gridSize, tooltipContent }: CovarianceCellProps) {
    const rowIndex = Math.floor(idx / gridSize);
    const colIndex = idx % gridSize;
    const borderClass = getCellBorderClass(value, absMax, rowIndex === colIndex);
    const displayPrecision = getAdaptivePrecision(value);
    const TooltipContent = tooltipContent;
    return (
        <Tooltip delayDuration={GRID_TOOLTIP_DELAY_DURATION}>
            <Tooltip.Trigger>
                <MatrixGrid.Cell className={borderClass}>
                    <div className="tabular-nums font-medium">
                        {value.toFixed(displayPrecision)}
                    </div>
                </MatrixGrid.Cell>
            </Tooltip.Trigger>
            <Tooltip.Content>
                <TooltipContent idx={idx} gridSize={gridSize} />
            </Tooltip.Content>
        </Tooltip>
    );
}

function getCellBorderClass(value: number, absMax: number, isDiagonal: boolean): string {
    if (absMax === 0) {
        return 'border-gray-300';
    }

    const intensity = absMax > 0 ? Math.abs(value) / absMax : 0;
    const percent = intensity * 100;

    if (isDiagonal) {
        // Diagonal → Green borders
        return getGreenBorderClass(percent);
    }

    if (value > 0) {
        // Positive → Red borders
        return getBlueBorderClass(percent);
    } else if (value < 0) {
        // Negative → Blue borders
        return getRedBorderClass(percent);
    }
    return 'border-gray-300';
}

const getRedBorderClass = (percent: number): string => {
    if (percent >= 80) return 'border-red-700';
    if (percent >= 60) return 'border-red-600';
    if (percent >= 40) return 'border-red-500';
    if (percent >= 20) return 'border-red-400';
    if (percent >= 10) return 'border-red-300';
    return 'border-red-200';
};

const getGreenBorderClass = (percent: number): string => {
    if (percent >= 80) return 'border-green-700';
    if (percent >= 60) return 'border-green-600';
    if (percent >= 40) return 'border-green-500';
    if (percent >= 20) return 'border-green-400';
    if (percent >= 10) return 'border-green-300';
    return 'border-green-200';
};

const getBlueBorderClass = (percent: number): string => {
    if (percent >= 80) return 'border-blue-700';
    if (percent >= 60) return 'border-blue-600';
    if (percent >= 40) return 'border-blue-500';
    if (percent >= 20) return 'border-blue-400';
    if (percent >= 10) return 'border-blue-300';
    return 'border-blue-200';
};

function getAdaptivePrecision(value: number): number {
    const absValue = Math.abs(value);
    if (absValue >= 1000) return 0;
    if (absValue >= 100) return 0;
    if (absValue >= 10) return 1;
    return 2;
}
