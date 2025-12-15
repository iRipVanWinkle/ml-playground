import { useContext } from 'react';
import { element } from '@/app/shared/helpers';
import { CovarianceMatrixContext } from './CovarianceMatrixContext';
import { Tooltip } from '@/app/shared/ui';
import { GRID_TOOLTIP_DELAY_DURATION } from '../../constants';

type TooltipWrapperProps = {
    rowIndex: number;
    colIndex: number;
    children: React.ReactNode;
};

export function TooltipWrapper({ rowIndex, colIndex, children }: TooltipWrapperProps) {
    return (
        <Tooltip delayDuration={GRID_TOOLTIP_DELAY_DURATION}>
            <Tooltip.Trigger>{children}</Tooltip.Trigger>
            <Tooltip.Content>
                <TooltipContent rowIndex={rowIndex} colIndex={colIndex} />
            </Tooltip.Content>
        </Tooltip>
    );
}

type TooltipContentProps = {
    rowIndex: number;
    colIndex: number;
};

export function TooltipContent({ rowIndex, colIndex }: TooltipContentProps) {
    const { covariances, featureLabels } = useContext(CovarianceMatrixContext);

    const isDiagonal = rowIndex === colIndex;

    if (isDiagonal) {
        const variance = element(covariances, rowIndex, colIndex);
        const stdDev = Math.sqrt(variance);
        return (
            <div className="text-sm flex flex-col gap-1">
                <div className="font-semibold">{featureLabels[rowIndex]}</div>
                <div>
                    Variance (σ²): <span className="font-medium">{variance.toFixed(6)}</span>
                </div>
                <div>
                    Std Dev (σ): <span className="font-medium">{stdDev.toFixed(6)}</span>
                </div>
            </div>
        );
    }

    const covariance = element(covariances, rowIndex, colIndex);
    const rowVariance = element(covariances, rowIndex, rowIndex);
    const colVariance = element(covariances, colIndex, colIndex);
    const correlation = covariance / Math.sqrt(rowVariance * colVariance);
    return (
        <div className="text-sm flex flex-col gap-1">
            <div className="font-semibold">
                {featureLabels[rowIndex]} vs {featureLabels[colIndex]}
            </div>
            <div>
                Covariance: <span className="font-medium">{covariance.toFixed(6)}</span>
            </div>
            <div>
                Correlation: <span className="font-medium">{correlation.toFixed(6)}</span>
            </div>
        </div>
    );
}
