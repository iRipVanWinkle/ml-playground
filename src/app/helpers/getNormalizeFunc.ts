import {
    linearScaling,
    logScaling,
    zScoreScaling,
    type NormalizatorFn,
} from '@/ml/data-processing/normalization';
import type { NormalizationFunction } from '@/app/store';

export function getNormalizeFunc(normalization: NormalizationFunction): NormalizatorFn | undefined {
    switch (normalization) {
        case 'zscore':
            return zScoreScaling;
        case 'linear':
            return linearScaling;
        case 'log':
            return logScaling;
        case 'none':
        default:
            return undefined; // No normalization
    }
}
