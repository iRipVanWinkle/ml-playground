import { createEmptyMatrix } from '@/app/shared/helpers';
import type { SettingsOf, TrainingReportOf } from '@/app/shared/registry';

export const DEFAULT_SETTINGS: SettingsOf<'k-means'> = {
    type: 'k-means',
    numClusters: 3,
    maxIterations: 100,
    centroidInitialization: { type: 'random' },
    distance: { type: 'euclidean' },
};

export const DEFAULT_REPORT: TrainingReportOf<'k-means'> = {
    type: 'k-means',
    taskType: 'clustering',
    iteration: 0,
    trainAssignments: createEmptyMatrix(),
    centroids: createEmptyMatrix(),
    inertiaHistory: [],
};
