import { GaussianNaiveBayes, QuadraticNaiveBayes } from '@/ml/models';
import type { TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { TrainingSettings } from '../../types';
import type { NaiveBayesSettings } from '../types';

export function naiveBayesModelFactory(
    settings: TrainingSettings<NaiveBayesSettings>,
    eventEmitter: TrainingEventEmitter,
    trainingController: TrainingControl,
) {
    const { modelSettings } = settings;

    let model;
    switch (modelSettings.variant) {
        case 'quadratic':
            model = new QuadraticNaiveBayes({ eventEmitter, trainingController });
            break;
        case 'gaussian':
        default:
            model = new GaussianNaiveBayes({ eventEmitter, trainingController });
            break;
    }

    return model;
}
