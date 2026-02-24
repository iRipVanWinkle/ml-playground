import { MinMaxScaler, LogScaler, ZScoreScaler } from '../../data-processing/normalization';
import type { NormalizationFunction } from './types';
import type { Scaler, ScalerParams } from '../../types';

export function normalizeFunctionFactory(
    normalization: NormalizationFunction,
): Scaler<ScalerParams> | undefined {
    switch (normalization) {
        case 'zscore':
            return new ZScoreScaler();
        case 'linear':
            return new MinMaxScaler();
        case 'log':
            return new LogScaler();
        case 'none':
        default:
            return undefined; // No normalization
    }
}
