import type {
    HierarchicalClusteringCallbackParameters as HierarchicalClusteringCallbackParametersML,
    HierarchicalClusteringParams,
} from '@/ml/types';
import type { BaseClusteringReport } from '@/app/shared/types';
import type { MatrixLike } from '@/app/shared/helpers';
import type { DistanceConfig } from '@/ml/factories';

export type HierarchicalMethod = 'divisive' | 'agglomerative';

export type Linkage = 'ward' | 'complete' | 'average' | 'single';

export type DivisiveClusteringSettings = {
    type: 'hierarchical';
    method: 'divisive';
    numClusters: number;
    bisectIterations: number;
    bisectRestarts: number;
    distance: DistanceConfig;
};

export type AgglomerativeClusteringSettings = {
    type: 'hierarchical';
    method: 'agglomerative';
    numClusters: number;
    linkage: Linkage;
    distance: DistanceConfig;
};

export type HierarchicalClusteringSettings =
    | DivisiveClusteringSettings
    | AgglomerativeClusteringSettings;

export type HierarchicalClusteringRepresentation = {
    type: 'hierarchical';
    representation: HierarchicalClusteringParams;
};

export type HierarchicalClusteringCallbackParameters = {
    type: 'hierarchical';
    callbackParameters: HierarchicalClusteringCallbackParametersML;
};

export type HierarchicalClusteringTrainingReport = BaseClusteringReport & {
    type: 'hierarchical';
    numClusters: number;
    trainAssignments: MatrixLike;
    testAssignments?: MatrixLike;
    params: HierarchicalClusteringParams | null;
};
