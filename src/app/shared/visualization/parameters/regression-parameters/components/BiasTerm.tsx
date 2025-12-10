type BiasTermProps = {
    bias: number;
    precision?: number;
};

export function BiasTerm({ bias, precision = 4 }: BiasTermProps) {
    return (
        <div className="flex items-center justify-between rounded-md p-3 bg-muted/80">
            <span className="text-sm font-medium text-muted-foreground">Intercept (Bias)</span>
            <span className="text-sm font-semibold">{bias.toFixed(precision)}</span>
        </div>
    );
}
