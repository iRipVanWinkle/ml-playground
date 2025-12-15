import { useColor } from '../../../colors';
import { FeatureGrid, FeatureBlock } from '../../../base';

const INITIAL_VISIBLE_COUNT = 10;

type MeanDisplayProps = {
    featureLabels: string[];
    means: number[];
};

export function MeanDisplay({ means, featureLabels }: MeanDisplayProps) {
    return (
        <FeatureBlock title="Means">
            <FeatureGrid
                items={means}
                labels={featureLabels}
                itemComponent={MeanItem}
                visibleCount={INITIAL_VISIBLE_COUNT}
            />
        </FeatureBlock>
    );
}

type MeanItemProps = {
    label: string;
    value: number;
    maxAbs: number;
};

export function MeanItem({ label, value, maxAbs }: MeanItemProps) {
    const { getColor } = useColor();

    const percentage = maxAbs > 0 ? (Math.abs(value) / maxAbs) * 100 : 0;
    const isPositive = value >= 0;
    const progressBarStyle = {
        width: `${percentage}%`,
        backgroundColor: isPositive ? getColor('red', 'lighten') : getColor('blue', 'lighten'),
    };

    return (
        <FeatureGrid.Cell label={label} progressStyle={progressBarStyle} withSign>
            {value}
        </FeatureGrid.Cell>
    );
}
