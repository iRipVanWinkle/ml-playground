import type { TrainingReport } from '@/app/models/types';
import type { CallbackParameters } from '@/ml/types';

export interface LiveMetrics<
    TCallbackParameters extends CallbackParameters,
    TTrainingReport extends TrainingReport,
> {
    updateIteration(params: TCallbackParameters): void;
    calculateMetrics(): Promise<TTrainingReport>;
    dispose?(): void;
}
