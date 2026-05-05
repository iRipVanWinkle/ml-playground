import type { NaiveBayesParams } from '@/ml/types';
import { row, type MatrixLike, type TypedArray } from '@/app/shared/helpers';
import { DEFAULT_PRECISION } from '@/app/shared/visualization/base';

type RawParametersProps = {
    params: NaiveBayesParams;
    categories: string[];
};

export function RawParameters({ params, categories }: RawParametersProps) {
    const numClasses = params.classes.length;
    const numFeatures = params.classMeans.shape[1];

    const formattedData: {
        classIndex: number;
        classLabel: string;
        prior: number;
        means: TypedArray;
        variances?: TypedArray;
        covariances?: MatrixLike;
    }[] = [];

    for (let c = 0; c < numClasses; c++) {
        const means = row(params.classMeans, c);
        const variances = params.type === 'gaussian' ? row(params.classVariances, c) : undefined;
        const covariances = params.type === 'quadratic' ? params.classCovariances[c] : undefined;
        formattedData.push({
            classIndex: c,
            classLabel: categories[params.classes[c]],
            prior: params.classPriors[c],
            means,
            variances,
            covariances,
        });
    }

    return (
        <div className="rounded-lg border bg-muted/50 p-4">
            <div className="text-sm text-muted-foreground text-center mb-6">
                Raw Parameters ({numClasses} class{numClasses > 1 ? 'es' : ''} × {numFeatures}{' '}
                features)
            </div>
            {formattedData.map(
                ({ classIndex, classLabel, prior, means, variances, covariances }) => (
                    <div
                        key={classIndex}
                        className="text-center flex flex-col gap-3 mb-6 last:mb-0"
                    >
                        <h3 className="text-lg font-bold text-center">Class {classLabel}</h3>
                        <Block title="Prior">{prior.toFixed(DEFAULT_PRECISION)}</Block>
                        <Block title="Means">{formatArray(means)}</Block>
                        {variances && <Block title="Variances">{formatArray(variances)}</Block>}
                        {covariances && (
                            <Block title="Covariance matrices">
                                <pre className="font-mono text-xs break-all leading-relaxed px-4 text-left overflow-auto max-h-64">
                                    {formatMatrix(covariances)}
                                </pre>
                            </Block>
                        )}
                    </div>
                ),
            )}
        </div>
    );
}

function Block({ children, title }: { children: React.ReactNode; title: string }) {
    return (
        <div>
            <p className="font-medium text-sm text-muted-foreground">{title}</p>
            <p className="font-mono text-xs leading-relaxed px-4">{children}</p>
        </div>
    );
}

const formatArray = (params: TypedArray) => {
    const values = Array.from(params).map((p) => p.toFixed(DEFAULT_PRECISION));
    return `[${values.join(', ')}]`;
};

const formatMatrix = (matrix: MatrixLike) => {
    const rows = Array.from({ length: matrix.shape[0] }, (_, r) => Array.from(row(matrix, r))).map(
        (row) => `[${row.map((v) => v.toFixed(DEFAULT_PRECISION)).join(', ')}]`,
    );
    return `[${rows.join(',\n ')}]`;
};
