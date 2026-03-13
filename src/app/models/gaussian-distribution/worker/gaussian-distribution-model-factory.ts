import { DiagonalGaussianDistribution, FullGaussianDistribution } from '@/ml/models';
import type { TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { TrainingSettings } from '../../types';
import type { GaussianDistributionSettings } from '../types';

export function gaussianDistributionModelFactory(
    settings: TrainingSettings<GaussianDistributionSettings>,
    eventEmitter?: TrainingEventEmitter,
    trainingController?: TrainingControl,
) {
    const { variant, threshold, varianceSmoothing } = settings.modelSettings;

    const options = { threshold, varianceSmoothing, eventEmitter, trainingController };

    if (variant === 'full') {
        return new FullGaussianDistribution(options);
    }

    return new DiagonalGaussianDistribution(options);
}
