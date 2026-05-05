import { createEmptyMatrix } from '@/app/shared/helpers';
import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';

export const DEFAULT_SETTINGS: SettingsOf<'dbscan'> = {
    type: 'dbscan',
    epsilon: 0.5,
    minPoints: 5,
    distance: { type: 'euclidean' },
};

export const DEFAULT_REPORT: TrainingReportOf<'dbscan'> = {
    type: 'dbscan',
    taskType: 'clustering',
    numClusters: 0,
    trainAssignments: createEmptyMatrix(),
    params: null,
};
