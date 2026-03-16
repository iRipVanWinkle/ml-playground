import { EMPTY_MATRIX_LIKE } from '@/app/shared/helpers';
import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';

export const DEFAULT_SETTINGS: SettingsOf<'hierarchical'> = {
    type: 'hierarchical',
    method: 'divisive',
    numClusters: 3,
    bisectIterations: 20,
    bisectRestarts: 3,
    distance: { type: 'euclidean' },
};

export const DEFAULT_REPORT: TrainingReportOf<'hierarchical'> = {
    type: 'hierarchical',
    taskType: 'clustering',
    numClusters: 0,
    trainAssignments: EMPTY_MATRIX_LIKE,
    params: null,
};
