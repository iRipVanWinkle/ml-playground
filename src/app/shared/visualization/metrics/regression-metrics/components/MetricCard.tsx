import { InfoTooltip } from '@/app/shared/ui';

interface MetricCardProps {
    label: string;
    value: number;
    tooltip: string;
    decimals?: number;
}

export function MetricCard({ label, value, tooltip, decimals = 4 }: MetricCardProps) {
    const formattedValue = Number.isFinite(value) ? value.toFixed(decimals) : 'N/A';

    return (
        <div className="p-4 rounded-lg text-center bg-primary-foreground">
            <div className="flex items-center justify-center gap-1 mb-1">
                <span className="text-muted-foreground text-sm font-medium">{label}</span>
                <InfoTooltip>{tooltip}</InfoTooltip>
            </div>
            <div className="text-xl font-semibold">{formattedValue}</div>
        </div>
    );
}
