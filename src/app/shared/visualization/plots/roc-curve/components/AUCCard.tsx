import { InfoTooltip } from '@/app/shared/ui';

interface AUCCardProps {
    label: string;
    value: number;
    tooltip?: string;
    className?: string;
    'data-testid'?: string;
}

export function AUCCard({ label, value, tooltip, className = '', ...rest }: AUCCardProps) {
    const testId = rest['data-testid'];

    return (
        <div className={`text-center ${className}`}>
            <div className="flex flex-row gap-1 items-center justify-center">
                <span className="text-muted-foreground text-xs">{label}</span>
                {tooltip && <InfoTooltip>{tooltip}</InfoTooltip>}
            </div>
            <div className="font-semibold" data-testid={testId ? `${testId}-value` : undefined}>
                {value.toFixed(3)}
            </div>
        </div>
    );
}
