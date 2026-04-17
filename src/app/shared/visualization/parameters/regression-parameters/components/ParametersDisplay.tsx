import { CategoryBlock, FeatureBlock, FeatureGrid, FeatureHighlight } from '../../../base';
import { useColor } from '../../../colors';

const INITIAL_VISIBLE_COUNT = 10;

type ParametersDisplayProps = {
    bias: number;
    weights: number[];
    categoryName?: string;
    featureLabels: string[];
    precision?: number;
};

export function ParametersDisplay({
    bias,
    weights,
    categoryName,
    featureLabels,
}: ParametersDisplayProps) {
    return (
        <CategoryBlock title={categoryName}>
            <FeatureHighlight label="Intercept (Bias)" data-testid="param-bias">
                {bias}
            </FeatureHighlight>
            <FeatureBlock title="Weights">
                <FeatureGrid
                    items={weights}
                    labels={featureLabels}
                    itemComponent={WeightItem}
                    visibleCount={INITIAL_VISIBLE_COUNT}
                />
            </FeatureBlock>
        </CategoryBlock>
    );
}

type WeightItemProps = {
    label: string;
    value: number;
    maxAbs: number;
};

function WeightItem({ label, value, maxAbs }: WeightItemProps) {
    const { getColor } = useColor();

    const percentage = (Math.abs(value) / maxAbs) * 100;
    const isPositive = value >= 0;
    const progressBarStyle = {
        width: `${percentage}%`,
        backgroundColor: isPositive ? getColor('red') : getColor('blue'),
    };

    return (
        <FeatureGrid.Cell label={label} progressStyle={progressBarStyle} withSign>
            {value}
        </FeatureGrid.Cell>
    );
}
