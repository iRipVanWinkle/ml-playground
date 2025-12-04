import { useMemo } from 'react';
import type {
    GaussianNaiveBayesParams,
    NaiveBayesParams,
    QuadraticNaiveBayesParams,
} from '@/ml/types';
import { useColor } from '../../../colors';

// Constants
const NUM_POINTS = 200;
const STD_MULTIPLIER = 4;

export interface PlotData {
    traces: Partial<Plotly.PlotData>[];
    featureName: string;
}

/**
 * Hook to calculate plot data based on model type
 */
export function useClassConditionalPlotData(
    params: NaiveBayesParams | undefined,
    categories: string[],
    headers: string[],
    featureIndex: number = 0,
): PlotData | null {
    const { getColor } = useColor();

    return useMemo(() => {
        if (params?.type === 'gaussian') {
            return calculateGaussianPlotData(params, categories, headers, featureIndex, getColor);
        } else if (params?.type === 'quadratic') {
            return calculateQuadraticPlotData(params, categories, headers, featureIndex, getColor);
        }
        return null;
    }, [params, categories, headers, featureIndex, getColor]);
}

/**
 * Calculate class-conditional distribution data for Gaussian Naive Bayes
 */
function calculateGaussianPlotData(
    params: GaussianNaiveBayesParams,
    categories: string[],
    headers: string[],
    featureIndex: number,
    getColor: (index: number) => string,
): PlotData {
    const { classes, classMeans, classVariances } = params;
    const featureName = headers[featureIndex + 1] || `Feature ${featureIndex}`;

    // Calculate x-axis range
    let minX = Infinity;
    let maxX = -Infinity;

    for (let c = 0; c < classes.length; c++) {
        const mean = classMeans[c][featureIndex];
        const std = Math.sqrt(classVariances[c][featureIndex]);
        minX = Math.min(minX, mean - STD_MULTIPLIER * std);
        maxX = Math.max(maxX, mean + STD_MULTIPLIER * std);
    }

    const xValues = Array.from(
        { length: NUM_POINTS },
        (_, i) => minX + (i / NUM_POINTS) * (maxX - minX),
    );

    // Calculate Gaussian PDF for each class
    const traces: Partial<Plotly.PlotData>[] = classes.map((cls, classIdx) => {
        const mean = classMeans[classIdx][featureIndex];
        const variance = classVariances[classIdx][featureIndex];
        const std = Math.sqrt(variance);

        const yValues = xValues.map((x) => {
            // Gaussian PDF: (1 / (σ√(2π))) * e^(-(x-μ)²/(2σ²))
            const exponent = -((x - mean) ** 2) / (2 * variance);
            return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
        });

        const categoryLabel = categories[cls] || `Class ${cls}`;

        return {
            x: xValues,
            y: yValues,
            mode: 'lines',
            name: `P(${featureName}|${categoryLabel})`,
            line: {
                color: getColor(classIdx),
            },
            type: 'scatter',
        };
    });

    return {
        traces,
        featureName,
    };
}

/**
 * Calculate class-conditional distribution data for Quadratic Naive Bayes
 */
function calculateQuadraticPlotData(
    params: QuadraticNaiveBayesParams,
    categories: string[],
    headers: string[],
    featureIndex: number,
    getColor: (index: number) => string,
): PlotData {
    const { classes, classMeans, classCovariances } = params;
    const featureName = headers[featureIndex + 1] || `Feature ${featureIndex}`;

    // Calculate x-axis range using diagonal of covariance matrix
    let minX = Infinity;
    let maxX = -Infinity;

    for (let c = 0; c < classes.length; c++) {
        const mean = classMeans[c][featureIndex];
        const variance = classCovariances[c][featureIndex][featureIndex];
        const std = Math.sqrt(Math.max(0, variance));
        minX = Math.min(minX, mean - STD_MULTIPLIER * std);
        maxX = Math.max(maxX, mean + STD_MULTIPLIER * std);
    }

    const xValues = Array.from(
        { length: NUM_POINTS },
        (_, i) => minX + (i / NUM_POINTS) * (maxX - minX),
    );

    // Calculate marginal Gaussian PDF for each class
    const traces: Partial<Plotly.PlotData>[] = classes.map((cls, classIdx) => {
        const mean = classMeans[classIdx][featureIndex];
        const variance = classCovariances[classIdx][featureIndex][featureIndex];
        const std = Math.sqrt(Math.max(1e-10, variance));

        const yValues = xValues.map((x) => {
            // Gaussian PDF: (1 / (σ√(2π))) * e^(-(x-μ)²/(2σ²))
            const exponent = -((x - mean) ** 2) / (2 * variance);
            return (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
        });

        const categoryLabel = categories[cls] || `Class ${cls}`;

        return {
            x: xValues,
            y: yValues,
            mode: 'lines',
            name: `P(${featureName}|${categoryLabel})`,
            line: {
                color: getColor(classIdx),
            },
            type: 'scatter',
        };
    });

    return {
        traces,
        featureName,
    };
}
