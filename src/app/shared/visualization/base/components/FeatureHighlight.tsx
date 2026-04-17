type FeatureHighlightProps = {
    label: string;
    children: number;
    precision?: number;
    'data-testid'?: string;
};

export function FeatureHighlight({
    label,
    children,
    precision = 4,
    ...rest
}: FeatureHighlightProps) {
    const testId = rest['data-testid'];

    return (
        <div className="flex items-center justify-between rounded-md p-3 bg-card">
            <span className="text-sm text-muted-foreground">{label}</span>
            <span
                className="text-sm font-semibold tabular-nums"
                data-testid={testId ? `${testId}-value` : undefined}
            >
                {children.toFixed(precision)}
            </span>
        </div>
    );
}
