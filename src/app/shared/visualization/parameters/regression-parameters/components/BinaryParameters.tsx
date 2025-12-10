import type { MatrixLike } from '@/app/shared/helpers';
import { ParametersDisplay } from './ParametersDisplay';

type BinaryParametersProps = {
    theta: MatrixLike;
    featureLabels: string[];
    precision?: number;
    isLinearRegression: boolean;
};

export function BinaryParameters({
    theta,
    featureLabels,
    precision = 4,
    isLinearRegression,
}: BinaryParametersProps) {
    const bias = theta.array[0];
    const weights = Array.from(theta.array.slice(1));

    const weightsEquation = weights.map((weight, index) => {
        const featureName = featureLabels[index] || `x${index + 1}`;
        const sign = weight >= 0 ? '+' : '';
        return ` ${sign} ${weight.toFixed(precision)} * ${featureName}`;
    });
    const equation = `${bias.toFixed(precision)}${weightsEquation.join('')}`;

    return (
        <>
            <ParametersDisplay
                bias={bias}
                weights={weights}
                featureLabels={featureLabels}
                precision={precision}
            />

            <div className="mt-6 rounded-lg border bg-muted/50 p-4">
                <div className="mb-2 text-sm font-medium text-muted-foreground">Model Equation</div>
                <div className="overflow-x-auto font-mono text-sm">
                    <div>{isLinearRegression ? `y = ${equation}` : `P(y=1) = σ(${equation})`}</div>
                </div>
            </div>
        </>
    );
}
