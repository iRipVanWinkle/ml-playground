import type { PredictionComponentProps } from '@/app/shared/registry/types';
import { RegressionPrediction } from './RegressionPrediction';
import { ClassificationPrediction } from './ClassificationPrediction';

export function TaskAwarePrediction(props: PredictionComponentProps) {
    return props.taskType === 'regression' ? (
        <RegressionPrediction {...props} />
    ) : (
        <ClassificationPrediction {...props} />
    );
}
