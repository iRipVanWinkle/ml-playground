import { KNNClassifier, KNNRegressor } from '@/ml/models';
import { distanceFactory } from '@/ml/factories';
import type { TrainingEventEmitter } from '@/ml/types';
import type { TrainingSettings } from '../../types';
import type { KNNSettings } from '../types';

export function knnModelFactory(
    settings: TrainingSettings<KNNSettings>,
    eventEmitter?: TrainingEventEmitter,
) {
    const { modelSettings, taskType } = settings;
    const { k, weights, distance } = modelSettings;

    const distanceMetric = distanceFactory(distance);

    if (taskType === 'regression') {
        return new KNNRegressor({ k, weights, distanceMetric, eventEmitter });
    }

    return new KNNClassifier({ k, weights, distanceMetric, eventEmitter });
}
