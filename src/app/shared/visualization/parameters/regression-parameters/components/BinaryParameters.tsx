import type { MatrixLike } from '@/app/shared/helpers';
import { ParametersDisplay } from './ParametersDisplay';

type BinaryParametersProps = {
    theta: MatrixLike;
    featureLabels: string[];
    precision?: number;
    isLinearRegression: boolean;
};

export function BinaryParameters({ theta, featureLabels, precision = 4 }: BinaryParametersProps) {
    const bias = theta.array[0];
    const weights = Array.from(theta.array.slice(1));

    return (
        <ParametersDisplay
            bias={bias}
            weights={weights}
            featureLabels={featureLabels}
            precision={precision}
        />
    );
}
