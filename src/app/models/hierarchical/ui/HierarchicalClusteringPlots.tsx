import type { Dataset } from '@/app/shared/types';
import type { HierarchicalClusteringTrainingReport } from '../types';
import { PlotlyScatter, PlotlyScatter3D } from '@/app/shared/visualization/plots/plotly';
import { useColor } from '@/app/shared/visualization/colors';

type Props = {
    dataset: Dataset;
    report: HierarchicalClusteringTrainingReport;
};

type ClusterBucket2D = { x: number[]; y: number[] };
type ClusterBucket3D = { x: number[]; y: number[]; z: number[] };

function buildClusterBuckets2D(
    features: number[][],
    assignments: number[],
): Record<number, ClusterBucket2D> {
    const buckets: Record<number, ClusterBucket2D> = {};
    for (let i = 0; i < features.length; i++) {
        const label = assignments[i];
        if (!buckets[label]) buckets[label] = { x: [], y: [] };
        buckets[label].x.push(features[i][0]);
        buckets[label].y.push(features[i][1]);
    }
    return buckets;
}

function buildClusterBuckets3D(
    features: number[][],
    assignments: number[],
): Record<number, ClusterBucket3D> {
    const buckets: Record<number, ClusterBucket3D> = {};
    for (let i = 0; i < features.length; i++) {
        const label = assignments[i];
        if (!buckets[label]) buckets[label] = { x: [], y: [], z: [] };
        buckets[label].x.push(features[i][0]);
        buckets[label].y.push(features[i][1]);
        buckets[label].z.push(features[i][2]);
    }
    return buckets;
}

export function HierarchicalClusteringPlots({ dataset, report }: Props) {
    const { trainInputFeatures, testInputFeatures, headers } = dataset;
    const { trainAssignments, testAssignments, numClusters } = report;
    const { getColor } = useColor();

    const is2DPlot = trainInputFeatures[0]?.length === 2;
    const is3DPlot = trainInputFeatures[0]?.length === 3;
    const hasAssignments = (trainAssignments?.shape[0] ?? 0) > 0;

    const [x1Label, x2Label, x3Label] = headers;

    const data2D = (() => {
        if (!is2DPlot) return [];

        if (!hasAssignments) {
            return [
                {
                    x: trainInputFeatures.map((p) => p[0]),
                    y: trainInputFeatures.map((p) => p[1]),
                    mode: 'markers' as const,
                    name: 'Training Data',
                    marker: { color: 'grey', size: 8 },
                },
                {
                    x: testInputFeatures.map((p) => p[0]),
                    y: testInputFeatures.map((p) => p[1]),
                    mode: 'markers' as const,
                    name: 'Test Data',
                    marker: { color: 'grey', size: 8, symbol: 'circle-open' as const },
                },
            ];
        }

        const trainBuckets = buildClusterBuckets2D(
            trainInputFeatures,
            Array.from(trainAssignments.array),
        );
        const testBuckets = testAssignments
            ? buildClusterBuckets2D(testInputFeatures, Array.from(testAssignments.array))
            : null;

        const traces = [];

        for (let c = 0; c < numClusters; c++) {
            if (trainBuckets[c]) {
                traces.push({
                    x: trainBuckets[c].x,
                    y: trainBuckets[c].y,
                    mode: 'markers' as const,
                    name: `Cluster ${c + 1}`,
                    marker: { color: getColor(c), size: 8, symbol: 'circle' as const },
                    legendgroup: `cluster-${c}`,
                });
            }
            if (testBuckets?.[c]) {
                traces.push({
                    x: testBuckets[c].x,
                    y: testBuckets[c].y,
                    mode: 'markers' as const,
                    name: `Cluster ${c + 1} (Test)`,
                    marker: { color: getColor(c), size: 8, symbol: 'circle-open' as const },
                    legendgroup: `cluster-${c}`,
                    showlegend: false,
                });
            }
        }

        return traces;
    })();

    const data3D = (() => {
        if (!is3DPlot) return [];

        if (!hasAssignments) {
            return [
                {
                    x: trainInputFeatures.map((p) => p[0]),
                    y: trainInputFeatures.map((p) => p[1]),
                    z: trainInputFeatures.map((p) => p[2]),
                    mode: 'markers' as const,
                    type: 'scatter3d' as const,
                    name: 'Training Data',
                    marker: { color: 'grey', size: 5 },
                },
                {
                    x: testInputFeatures.map((p) => p[0]),
                    y: testInputFeatures.map((p) => p[1]),
                    z: testInputFeatures.map((p) => p[2]),
                    mode: 'markers' as const,
                    type: 'scatter3d' as const,
                    name: 'Test Data',
                    marker: { color: 'grey', size: 5, symbol: 'circle-open' as const },
                },
            ];
        }

        const trainBuckets = buildClusterBuckets3D(
            trainInputFeatures,
            Array.from(trainAssignments.array),
        );
        const testBuckets = testAssignments
            ? buildClusterBuckets3D(testInputFeatures, Array.from(testAssignments.array))
            : null;

        const traces = [];

        for (let c = 0; c < numClusters; c++) {
            if (trainBuckets[c]) {
                traces.push({
                    x: trainBuckets[c].x,
                    y: trainBuckets[c].y,
                    z: trainBuckets[c].z,
                    mode: 'markers' as const,
                    type: 'scatter3d' as const,
                    name: `Cluster ${c + 1}`,
                    marker: { color: getColor(c), size: 5 },
                    legendgroup: `cluster-${c}`,
                });
            }
            if (testBuckets?.[c]) {
                traces.push({
                    x: testBuckets[c].x,
                    y: testBuckets[c].y,
                    z: testBuckets[c].z,
                    mode: 'markers' as const,
                    type: 'scatter3d' as const,
                    name: `Cluster ${c + 1} (Test)`,
                    marker: { color: getColor(c), size: 5, symbol: 'circle-open' as const },
                    legendgroup: `cluster-${c}`,
                    showlegend: false,
                });
            }
        }

        return traces;
    })();

    if (is2DPlot) {
        return (
            <PlotlyScatter
                data={data2D}
                layout={{
                    title: { text: 'Divisive Clustering' },
                    xaxis: { title: { text: x1Label } },
                    yaxis: { title: { text: x2Label } },
                }}
            />
        );
    }

    if (is3DPlot) {
        return (
            <PlotlyScatter3D
                data={data3D}
                layout={{
                    title: { text: 'Divisive Clustering (3D)' },
                    scene: {
                        xaxis: { title: { text: x1Label } },
                        yaxis: { title: { text: x2Label } },
                        zaxis: { title: { text: x3Label } },
                    },
                }}
            />
        );
    }

    return (
        <p className="text-sm text-muted-foreground p-4">Plotting requires 2 or 3 input features</p>
    );
}
