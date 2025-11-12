import { InfoTooltip } from '@/app/shared/ui';
import { formatDecimal, formatPercentage } from '../utils';

interface MetricCardProps {
    label: string;
    value: number | null;
    tooltip: string;
    format: 'percentage' | 'decimal';
    decimals?: number;
}

export function MetricCard({ label, value, tooltip, format, decimals }: MetricCardProps) {
    const formattedValue =
        value === null
            ? 'N/A'
            : format === 'percentage'
              ? formatPercentage(value)
              : formatDecimal(value, decimals);

    return (
        <div className="text-center">
            <div>
                <span className="text-muted-foreground text-xs">{label}</span>{' '}
                <InfoTooltip>{tooltip}</InfoTooltip>
            </div>
            <div className="font-semibold">{formattedValue}</div>
        </div>
    );
}
