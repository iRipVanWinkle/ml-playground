import { useMemo } from 'react';
import type { TrainingReport } from '@/app/models/types';
import type { ParametersVisualizationProps } from '@/app/shared/registry';
import type { Dataset } from '@/app/shared/types';
import { ClusterCentroid } from './components';
import type { MatrixLike } from '@/app/shared/helpers';

export function KMeansParameters({
    report,
    dataset,
}: ParametersVisualizationProps<TrainingReport>) {
    const supportsKMeansParameters = report.type === 'k-means';

    if (!supportsKMeansParameters) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-center text-muted-foreground">
                    This model does not support K-Means parameters.
                </div>
            </div>
        );
    }

    if (report.centroids.shape[0] === 0) {
        return (
            <div className="w-full h-full p-4 flex items-center justify-center">
                <div className="text-sm text-center text-muted-foreground">
                    Run training to see cluster centroids.
                </div>
            </div>
        );
    }

    return <KMeansParametersContent centroids={report.centroids} dataset={dataset} />;
}

interface KMeansParametersContentProps {
    centroids: MatrixLike;
    dataset: Dataset;
}

function KMeansParametersContent({ centroids, dataset }: KMeansParametersContentProps) {
    const [numClusters, numFeatures] = centroids.shape;

    const featureLabels = useMemo(() => {
        return dataset.headers.length >= numFeatures
            ? dataset.headers.slice(0, numFeatures)
            : Array.from({ length: numFeatures }, (_, i) => `Feature ${i + 1}`);
    }, [dataset.headers, numFeatures]);

    const clusterCentroids = useMemo(() => {
        const clusters: number[][] = [];
        for (let c = 0; c < numClusters; c++) {
            const rowStart = c * numFeatures;
            const centroid: number[] = [];
            for (let f = 0; f < numFeatures; f++) {
                centroid.push(centroids.array[rowStart + f]);
            }
            clusters.push(centroid);
        }
        return clusters;
    }, [centroids, numClusters, numFeatures]);

    return (
        <div className="w-full grid grid-cols-1 gap-3 p-4">
            <h3 className="mb-4 text-lg font-semibold">Cluster Centroids</h3>
            <div className="flex flex-col gap-3">
                {clusterCentroids.map((centroid, index) => (
                    <ClusterCentroid
                        key={index}
                        clusterIndex={index}
                        centroid={centroid}
                        featureLabels={featureLabels}
                    />
                ))}
            </div>
        </div>
    );
}
