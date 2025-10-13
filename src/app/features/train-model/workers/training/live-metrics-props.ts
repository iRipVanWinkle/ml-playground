import type { ModelSettings } from '@/app/features/configure-model';
import type { DataState } from '@/app/features/load-dataset';
import type { TaskType } from '@/app/shared/types';

export class LiveMetricsProps {
    readonly isOneVsRest: boolean;
    readonly taskType: TaskType;
    readonly numThreads: number;

    get isClassificationTask(): boolean {
        return this.taskType === 'classification';
    }

    constructor(state: { taskType: TaskType; data: DataState; modelSettings: ModelSettings }) {
        const { modelSettings, taskType, data } = state;

        this.isOneVsRest =
            modelSettings.type === 'logistic' && modelSettings.classificationType === 'ovr';

        this.taskType = taskType;

        this.numThreads = 1;
        if (this.isOneVsRest) {
            this.numThreads = data.categories?.length ?? 1;
        }
    }
}
