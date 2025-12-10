import { useColor } from '../../../colors';

type WeightItemProps = {
    featureName: string;
    weight: number;
    maxAbsWeight: number;
    precision?: number;
};

export function WeightItem({ featureName, weight, maxAbsWeight, precision = 4 }: WeightItemProps) {
    const { getColor } = useColor();

    const percentage = (Math.abs(weight) / maxAbsWeight) * 100;
    const isPositive = weight >= 0;
    const progressBarStyle = {
        width: `${percentage}%`,
        backgroundColor: isPositive ? getColor('green') : getColor('red'),
    };

    return (
        <div className="flex items-center p-2 pt-1 pb-3 border-b">
            <div
                className="flex-1 truncate text-left text-sm font-medium text-muted-foreground"
                title={featureName}
            >
                {featureName}
            </div>
            <div className="ml-3 flex shrink-0 items-center gap-2">
                <div className="h-2 w-16 overflow-hidden rounded-full bg-muted">
                    <div className="h-full rounded-full transition-all" style={progressBarStyle} />
                </div>
                <div className="w-16 text-right text-sm">
                    {isPositive ? '+' : ''}
                    {weight.toFixed(precision)}
                </div>
            </div>
        </div>
    );
}
