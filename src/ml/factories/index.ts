// Factory functions
export { normalizeFunctionFactory } from './normalization/factory';
export { transformationsFactory } from './transformations/factory';
export { regularizationFactory } from './regularization/factory';
export { thetaInitializerFactory } from './theta-initialization/factory';
export { criterionFactory } from './criterion/factory';
export { lossFunctionFactory } from './loss-function/factory';
export { optimizerFactory } from './optimizer/factory';

// Types
export type { NormalizationFunction } from './normalization/types';
export type { TransformationFunction, TransformationConfig } from './transformations/types';
export type { RegularizationConfig, RegularizationType } from './regularization/types';
export type { ThetaInitializationConfig } from './theta-initialization/types';
export type { CriterionConfig, CriterionType } from './criterion/types';
export type { LossFunctionConfig, LossFunctionType } from './loss-function/types';
export type { OptimizerConfig } from './optimizer/types';
