type FeatureHighlightProps = {
    label: string;
    children: number;
    precision?: number;
};

export function FeatureHighlight({ label, children, precision = 4 }: FeatureHighlightProps) {
    return (
        <div className="flex items-center justify-between rounded-md p-3 bg-card">
            <span className="text-sm text-muted-foreground">{label}</span>
            <span className="text-sm font-semibold tabular-nums">
                {children.toFixed(precision)}
            </span>
        </div>
    );
}
