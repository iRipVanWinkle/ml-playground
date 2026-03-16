import type { HierarchicalClusteringTrainingReport } from '../types';
import type { MainMetricsProps } from '@/app/shared/registry';

export function HierarchicalClusteringMainMetrics({
    report,
}: MainMetricsProps<HierarchicalClusteringTrainingReport>) {
    const { numClusters } = report;

    return (
        <div>
            Clusters: <div className="font-bold tabular-nums">{numClusters ?? 0}</div>
        </div>
    );
}
