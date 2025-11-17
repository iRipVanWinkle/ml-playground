import type { RocCurveData } from '../types';

/**
 * Checks if the ROC curve data contains valid data for visualization.
 * For multiclass: checks if there are any AUC scores.
 * For binary: checks if there are any thresholds.
 */
export function hasRocCurveData(rocCurveData: RocCurveData): boolean {
    return (
        (rocCurveData.type === 'multiclass' && rocCurveData.aucs.length > 0) ||
        (rocCurveData.type === 'binary' && rocCurveData.thresholds.length > 0)
    );
}
