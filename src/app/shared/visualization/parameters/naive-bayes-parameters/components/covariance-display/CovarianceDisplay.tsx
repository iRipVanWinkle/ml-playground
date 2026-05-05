import { createContext, useContext } from 'react';
import { element, createEmptyMatrix, type MatrixLike } from '@/app/shared/helpers';
import { FeatureBlock, GRID_VIEW_THRESHOLD, ImageGrid } from '@/app/shared/visualization/base';
import { CovarianceMatrixGrid } from './CovarianceMatrixGrid';

const CovarianceMatrixContext = createContext<{
    covariances: MatrixLike;
    featureLabels: string[];
}>({
    covariances: createEmptyMatrix(),
    featureLabels: [],
});

type CovarianceDisplayProps = {
    featureLabels: string[];
    covariances: MatrixLike;
    precision?: number;
};

export function CovarianceDisplay({ covariances, featureLabels }: CovarianceDisplayProps) {
    const gridSize = covariances.shape[0];
    const useGridView = gridSize <= GRID_VIEW_THRESHOLD;

    const contextValue = { covariances, featureLabels };

    return (
        <FeatureBlock title="Covariance Matrix">
            <CovarianceMatrixContext.Provider value={contextValue}>
                {useGridView ? (
                    <CovarianceMatrixGrid
                        covariances={covariances}
                        featureLabels={featureLabels}
                        tooltipContent={TooltipContent}
                    />
                ) : (
                    <ImageGrid
                        values={covariances.array}
                        gridSize={gridSize}
                        tooltipContent={TooltipContent}
                    />
                )}
            </CovarianceMatrixContext.Provider>
        </FeatureBlock>
    );
}

type TooltipContentProps = {
    idx: number;
    gridSize: number;
};

export function TooltipContent({ idx, gridSize }: TooltipContentProps) {
    const { covariances, featureLabels } = useContext(CovarianceMatrixContext);

    const rowIndex = Math.floor(idx / gridSize);
    const colIndex = idx % gridSize;

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
