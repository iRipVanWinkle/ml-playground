import { IsolationForest } from '@/ml/models';
import type { TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { TrainingSettings } from '../../types';
import type { IsolationForestSettings } from '../types';

export function isolationForestModelFactory(
    settings: TrainingSettings<IsolationForestSettings>,
    eventEmitter?: TrainingEventEmitter,
    trainingController?: TrainingControl,
) {
    const { estimators, maxSamples, contamination, bootstrap } = settings.modelSettings;

    return new IsolationForest({
        estimators,
        maxSamples,
        contamination,
        bootstrap,
        eventEmitter,
        trainingController,
    });
}
