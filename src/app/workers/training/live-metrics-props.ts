import type { State, TaskType } from '../../store';

export class LiveMetricsProps {
    readonly isOneVsRest: boolean;
    readonly taskType: TaskType;
    readonly numThreads: number;

    get isClassificationTask(): boolean {
        return this.taskType === 'classification';
    }

    constructor(state: State) {
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
