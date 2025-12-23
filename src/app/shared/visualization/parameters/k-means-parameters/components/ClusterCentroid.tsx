import { CategoryBlock, FeatureBlock, FeatureGrid } from '../../../base';

type ClusterCentroidProps = {
    clusterIndex: number;
    centroid: number[];
    featureLabels: string[];
};

export function ClusterCentroid({ clusterIndex, centroid, featureLabels }: ClusterCentroidProps) {
    const clusterLabel = `Cluster ${clusterIndex}`;

    return (
        <CategoryBlock title={clusterLabel}>
            <FeatureBlock title="Centroid Coordinates">
                <FeatureGrid items={centroid} labels={featureLabels} itemComponent={CentroidItem} />
            </FeatureBlock>
        </CategoryBlock>
    );
}

type CentroidItemProps = {
    label: string;
    value: number;
};

function CentroidItem({ label, value }: CentroidItemProps) {
    return (
        <div className="flex items-center px-2 pb-3 border-b border-border/50">
            <div className="flex-1 truncate text-left text-sm text-muted-foreground" title={label}>
                {label}
            </div>
            <div className="text-sm font-medium tabular-nums">{value.toFixed(4)}</div>
        </div>
    );
}
