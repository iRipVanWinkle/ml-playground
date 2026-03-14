import { DBSCAN } from '@/ml/models/dbscan/DBSCAN';
import { distanceFactory } from '@/ml/factories';
import type { TrainingControl, TrainingEventEmitter } from '@/ml/types';
import type { TrainingSettings } from '../../types';
import type { DBSCANSettings } from '../types';

export function dbscanModelFactory(
    settings: TrainingSettings<DBSCANSettings>,
    eventEmitter?: TrainingEventEmitter,
    trainingController?: TrainingControl,
) {
    const { epsilon, minPoints, distance } = settings.modelSettings;
    const distanceMetric = distanceFactory(distance);

    return new DBSCAN({ epsilon, minPoints, distanceMetric, eventEmitter, trainingController });
}
