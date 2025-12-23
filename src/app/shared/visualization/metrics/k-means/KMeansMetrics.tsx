import { useEffect, useMemo, useState } from 'react';
import type { TrainingReport } from '@/app/models/types';
import type { Dataset } from '@/app/shared/types';
import type { KMeansMetricsData } from './type';
import { CategoryBlock, FeatureGrid } from '../../base';
import { TrainTestSelector } from '@/app/shared/ui';

interface KMeansMetricsProps {
    dataset: Dataset;
    report: TrainingReport;
}

export function KMeansMetrics({ dataset, report }: KMeansMetricsProps) {
    const [selectedDataset, setSelectedDataset] = useState<string>('train');

    const supportsKMeansMetrics = report.type === 'k-means';
    const hasKMeansMetrics = supportsKMeansMetrics && report.trainMetrics != null;

    useEffect(() => {
        // This is a workaround to avoid the issue: Calling setState synchronously within an effect can trigger cascading renders
        setTimeout(() => {
            setSelectedDataset('train');
        }, 0);
    }, [dataset]);

    if (!supportsKMeansMetrics) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                    The model does not support K-Means metrics.
                </div>
            </div>
        );
    }

    if (!hasKMeansMetrics) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                    Run training to see K-Means metrics
                </div>
            </div>
        );
    }

    const kMeansMetrics = selectedDataset === 'train' ? report.trainMetrics! : report.testMetrics!;

    return (
        <div className="w-full bg-card h-full p-4 flex flex-col gap-4">
            <div className="flex flex-row justify-end">
                <div className="flex flex-row gap-2">
                    {report.testMetrics && (
                        <TrainTestSelector value={selectedDataset} onChange={setSelectedDataset} />
                    )}
                </div>
            </div>

            <KMeansMetricsContent metrics={kMeansMetrics} />
        </div>
    );
}

type KMeansMetricsContentProps = {
    metrics: KMeansMetricsData;
};

function KMeansMetricsContent({ metrics }: KMeansMetricsContentProps) {
    const preparedMetrics = useMemo(() => {
        return Array.from(metrics.silhouetteClusterScores).map((silhouetteScore, clusterId) => {
            const clusterLabel = `Cluster ${clusterId}`;

            const radius = metrics.maxDistanceToCenter[clusterId];
            const cohesion = metrics.avgDistanceToCenter[clusterId];
            const separation = metrics.avgDistanceToOtherCenters[clusterId];

            return {
                clusterLabel,
                metrics: [silhouetteScore, radius, cohesion, separation],
                labels: ['Silhouette Score', 'Radius', 'Cohesion', 'Separation'],
            };
        });
    }, [metrics]);

    return (
        <div className="w-full grid grid-cols-2 gap-3">
            {preparedMetrics.map(({ clusterLabel, metrics, labels }) => (
                <CategoryBlock key={clusterLabel} title={clusterLabel}>
                    <FeatureGrid oneColumn items={metrics} labels={labels} itemComponent={Item} />
                </CategoryBlock>
            ))}
        </div>
    );
}

function Item({ label, value }: { label: string; value: number }) {
    return (
        <div className="flex items-center px-2 pb-3 border-b border-border/50">
            <div className="flex-1 truncate text-left text-sm text-muted-foreground" title={label}>
                {label}
            </div>
            <div className="ml-3 shrink-0 text-right text-sm font-medium tabular-nums">
                {value.toFixed(4)}
            </div>
        </div>
    );
}
