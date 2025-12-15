import type { NaiveBayesParams } from '@/ml/types';
import { row } from '@/app/shared/helpers';
import { CategoryBlock, FeatureHighlight } from '../../../base';
import { VarianceDisplay } from './VarianceDisplay';
import { MeanDisplay } from './MeanDisplay';
import { CovarianceDisplay } from './covariance-display';

type TabularParametersProps = {
    params: NaiveBayesParams;
    featureLabels: string[];
    categories?: string[];
    selectedClassIndex?: number;
};

export function TabularParameters({
    params,
    featureLabels,
    categories,
    selectedClassIndex,
}: TabularParametersProps) {
    const classes = selectedClassIndex !== undefined ? [selectedClassIndex] : params.classes;

    return (
        <div className="flex flex-col gap-3">
            {classes.map((classIndex) => {
                const categoryName = categories?.[classIndex] || `Class ${classIndex}`;
                return (
                    <ParametersDisplay
                        key={classIndex}
                        categoryName={categoryName}
                        params={params}
                        classIndex={classIndex}
                        featureLabels={featureLabels}
                    />
                );
            })}
        </div>
    );
}

type ParametersDisplayProps = {
    params: NaiveBayesParams;
    categoryName?: string;
    classIndex: number;
    featureLabels: string[];
};

export function ParametersDisplay({
    params,
    categoryName,
    classIndex,
    featureLabels,
}: ParametersDisplayProps) {
    const prior = params.classPriors[classIndex];
    const means = Array.from(row(params.classMeans, classIndex));

    let variances = undefined;
    if (params.type === 'gaussian') {
        variances = Array.from(row(params.classVariances, classIndex));
    }

    let covariances = undefined;
    if (params.type === 'quadratic') {
        covariances = params.classCovariances[classIndex];
    }

    return (
        <CategoryBlock title={categoryName || `Class ${classIndex}`}>
            <FeatureHighlight label="Prior">{prior}</FeatureHighlight>

            <MeanDisplay means={means} featureLabels={featureLabels} />

            {variances && <VarianceDisplay variances={variances} featureLabels={featureLabels} />}
            {covariances && (
                <CovarianceDisplay covariances={covariances} featureLabels={featureLabels} />
            )}
        </CategoryBlock>
    );
}
