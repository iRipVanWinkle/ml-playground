import type { ReactNode } from 'react';
import { Tooltip } from '@/app/shared/ui';
import { getPercentage } from '../utils';

interface MatrixCellProps {
    value: number;
    rowTotal: number;
    tooltip: ReactNode;
    isDiagonal: boolean;
}

const DELAY_DURATION = 500;

export function MatrixCell({ value, rowTotal, tooltip, isDiagonal }: MatrixCellProps) {
    const percentage = getPercentage(value, rowTotal);

    const borderClass = isDiagonal
        ? getGreenBorderClass(percentage)
        : value === 0
          ? 'border-gray-300'
          : getRedBorderClass(percentage);

    return (
        <Tooltip delayDuration={DELAY_DURATION}>
            <Tooltip.Trigger>
                <div
                    data-type="matrix-cell"
                    className={`
                flex flex-col items-center justify-center
                text-sm font-medium
                aspect-square w-full min-w-[50px] min-h-[50px] max-w-[60px] max-h-[60px]
                border-2 box-border cursor-pointer transition-all duration-200
                hover:scale-105
                ${borderClass}
            `}
                >
                    <>
                        <div className="text-base font-semibold">{value}</div>
                        <div className="text-xs text-muted-foreground">{percentage}%</div>
                    </>
                </div>
            </Tooltip.Trigger>
            <Tooltip.Content>{tooltip}</Tooltip.Content>
        </Tooltip>
    );
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
    if (percent >= 80) return 'border-green-600';
    if (percent >= 50) return 'border-green-500';
    if (percent >= 20) return 'border-green-400';
    return 'border-green-300';
};
