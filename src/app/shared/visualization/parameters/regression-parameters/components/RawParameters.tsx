import type { MatrixLike } from '@/app/shared/helpers';

type RawParametersProps = {
    theta: MatrixLike;
    precision?: number;
};

export function RawParameters({ theta, precision = 6 }: RawParametersProps) {
    const [numClasses, numFeatures] = theta.shape;

    const formattedRows: string[][] = [];
    for (let c = 0; c < numClasses; c++) {
        const rowStart = c * numFeatures;
        const row: string[] = [];
        for (let f = 0; f < numFeatures; f++) {
            row.push(theta.array[rowStart + f].toFixed(precision));
        }
        formattedRows.push(row);
    }

    return (
        <div className="rounded-lg border bg-muted/50 p-4">
            <div className="mb-3 text-sm font-medium text-muted-foreground">
                Raw Parameters ({numClasses} class{numClasses > 1 ? 'es' : ''} × {numFeatures}{' '}
                features)
            </div>
            <div className="overflow-x-auto">
                <pre className="font-mono text-xs leading-relaxed">
                    {formattedRows.map((row, idx) => (
                        <div key={idx} className="whitespace-nowrap">
                            [{row.join(', ')}]
                        </div>
                    ))}
                </pre>
            </div>
        </div>
    );
}
