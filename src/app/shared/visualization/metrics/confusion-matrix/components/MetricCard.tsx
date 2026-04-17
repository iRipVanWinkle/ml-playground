import { InfoTooltip } from '@/app/shared/ui';
import { formatDecimal, formatPercentage } from '../utils';

interface MetricCardProps {
    label: string;
    value: number | null;
    tooltip: string;
    format: 'percentage' | 'decimal';
    decimals?: number;
    'data-testid'?: string;
}

export function MetricCard({ label, value, tooltip, format, decimals, ...rest }: MetricCardProps) {
    const testId = rest['data-testid'];
    const formattedValue =
        value === null
            ? 'N/A'
            : format === 'percentage'
              ? formatPercentage(value)
              : formatDecimal(value, decimals);

    return (
        <div className="text-center">
            <div>
                <div className="flex flex-row gap-1.5 items-center justify-center">
                    <span className="text-muted-foreground text-xs">{label}</span>
                    <InfoTooltip>{tooltip}</InfoTooltip>
                </div>
            </div>
            <div className="font-semibold" data-testid={testId ? `${testId}-value` : undefined}>
                {formattedValue}
            </div>
        </div>
    );
}
