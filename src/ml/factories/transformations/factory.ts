import {
    cosinusoidGenerator,
    fourierGenerator,
    fullPolynomialGenerator,
    sinusoidGenerator,
    type TransformationFn,
} from '../../data-processing/transformation';
import type { TransformationFunction } from './types';

export function transformationsFactory(
    transformationsConfig: {
        type: TransformationFunction;
        degree: number;
    }[],
): TransformationFn[] {
    const transformations = [];
    for (const transformation of transformationsConfig) {
        const { type, degree } = transformation;
        switch (type) {
            case 'sinusoid':
                transformations.push(sinusoidGenerator(degree));
                break;
            case 'cosinusoid':
                transformations.push(cosinusoidGenerator(degree));
                break;
            case 'fourier':
                transformations.push(fourierGenerator(degree));
                break;
            case 'polynomial':
                transformations.push(fullPolynomialGenerator(degree));
                break;
            default:
                console.warn(`Unknown transformation type: ${transformation.type}`);
        }
    }

    return transformations;
}
