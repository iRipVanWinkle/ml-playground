import type { Model, ModelRepresentation } from '@/ml/types';
import { PreprocessingModelDecorator } from '@/ml/models';
import { EventEmitter } from '@/ml/events/EventEmitter';
import type { TransformationSettings } from '../../transform-data';
import { normalizeFunctionFactory, transformationsFactory } from '@/ml/factories';

export function createPreprocessingPipeline(
    model: Model<ModelRepresentation>,
    dataSettings: TransformationSettings,
    eventEmitter: EventEmitter,
): PreprocessingModelDecorator<ModelRepresentation> {
    const normalizeFunction = normalizeFunctionFactory(dataSettings.normalization);
    const transformations = transformationsFactory(dataSettings.transformations, normalizeFunction);

    const featureTransform = {
        normalizeFunction,
        transformations,
    };

    return new PreprocessingModelDecorator(model, featureTransform, eventEmitter);
}
