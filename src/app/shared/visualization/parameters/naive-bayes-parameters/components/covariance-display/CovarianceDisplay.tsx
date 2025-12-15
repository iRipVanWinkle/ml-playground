import { CovarianceMatrixGrid } from './CovarianceMatrixGrid';
import { FeatureBlock } from '../../../../base';
import { CovarianceMatrixHeatmap } from './CovarianceMatrixHeatmap';
import type { MatrixLike } from '@/app/shared/helpers';
import { GRID_VIEW_THRESHOLD } from '../../constants';
import { CovarianceMatrixContext } from './CovarianceMatrixContext';
import { useMemo } from 'react';

type CovarianceDisplayProps = {
    featureLabels: string[];
    covariances: MatrixLike;
    precision?: number;
};

export function CovarianceDisplay({ covariances, featureLabels }: CovarianceDisplayProps) {
    const gridSize = covariances.shape[0];
    const useGridView = gridSize <= GRID_VIEW_THRESHOLD;

    const contextValue = useMemo(
        () => ({ covariances, featureLabels }),
        [covariances, featureLabels],
    );

    return (
        <FeatureBlock title="Covariance Matrix">
            <CovarianceMatrixContext.Provider value={contextValue}>
                {useGridView ? (
                    <CovarianceMatrixGrid covariances={covariances} featureLabels={featureLabels} />
                ) : (
                    <CovarianceMatrixHeatmap covariances={covariances} />
                )}
            </CovarianceMatrixContext.Provider>
        </FeatureBlock>
    );
}
