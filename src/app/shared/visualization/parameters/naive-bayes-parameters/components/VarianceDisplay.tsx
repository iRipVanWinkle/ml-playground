import { useColor } from '../../../colors';
import { FeatureBlock, FeatureGrid } from '../../../base';

type VarianceDisplayProps = {
    featureLabels: string[];
    variances: number[];
};

export function VarianceDisplay({ variances, featureLabels }: VarianceDisplayProps) {
    return (
        <FeatureBlock title="Variances">
            <FeatureGrid items={variances} labels={featureLabels} itemComponent={VarianceItem} />
        </FeatureBlock>
    );
}

type VarianceItemProps = {
    label: string;
    value: number;
    maxAbs: number;
};

export function VarianceItem({ label, value, maxAbs }: VarianceItemProps) {
    const { getColor } = useColor();

    const percentage = maxAbs > 0 ? (Math.abs(value) / maxAbs) * 100 : 0;
    const progressBarStyle = {
        width: `${percentage}%`,
        backgroundColor: getColor('green', 'lighten'),
    };

    return (
        <FeatureGrid.Cell label={label} progressStyle={progressBarStyle}>
            {value}
        </FeatureGrid.Cell>
    );
}
