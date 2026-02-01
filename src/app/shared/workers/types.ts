import type { TrainingReport } from '@/app/models/types';
import type { CallbackParameters } from '@/ml/types';

export interface LiveMetrics<
    TCallbackParameters extends CallbackParameters,
    TTrainingReport extends TrainingReport,
> {
    calculateMetrics(params: TCallbackParameters): Promise<TTrainingReport>;
    dispose?(): void;
}
