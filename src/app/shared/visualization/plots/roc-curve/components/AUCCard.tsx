import { InfoTooltip } from '@/app/shared/ui';

interface AUCCardProps {
    label: string;
    value: number;
    tooltip?: string;
    className?: string;
}

export function AUCCard({ label, value, tooltip, className = '' }: AUCCardProps) {
    return (
        <div className={`text-center ${className}`}>
            <div className="flex flex-row gap-1 items-center justify-center">
                <span className="text-muted-foreground text-xs">{label}</span>
                {tooltip && <InfoTooltip>{tooltip}</InfoTooltip>}
            </div>
            <div className="font-semibold">{value.toFixed(3)}</div>
        </div>
    );
}
