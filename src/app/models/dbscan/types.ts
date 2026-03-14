import type {
    DBSCANCallbackParameters as DBSCANCallbackParametersMl,
    DBSCANParams,
} from '@/ml/types';
import type { BaseClusteringReport } from '@/app/shared/types';
import type { MatrixLike } from '@/app/shared/helpers';
import type { DistanceConfig } from '@/ml/factories';

export type DBSCANSettings = {
    type: 'dbscan';
    epsilon: number;
    minPoints: number;
    distance: DistanceConfig;
};

export type DBSCANRepresentation = {
    type: 'dbscan';
    representation: DBSCANParams;
};

export type DBSCANCallbackParameters = {
    type: 'dbscan';
    callbackParameters: DBSCANCallbackParametersMl;
};

export type DBSCANTrainingReport = BaseClusteringReport & {
    type: 'dbscan';
    numClusters: number;
    activePointIndex?: number;
    trainAssignments: MatrixLike;
    testAssignments?: MatrixLike;
    params: DBSCANParams | null;
    trainSilhouetteScore?: number;
    testSilhouetteScore?: number;
};
