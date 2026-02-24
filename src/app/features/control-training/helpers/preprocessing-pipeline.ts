import type { Model, ModelRepresentation } from '@/ml/types';
import { PipelineModel } from '@/ml/models';
import { EventEmitter } from '@/ml/events/EventEmitter';
import type { TransformationSettings } from '../../transform-data';
import { normalizeFunctionFactory, transformationsFactory } from '@/ml/factories';

export function createPreprocessingPipeline(
    model: Model<ModelRepresentation>,
    dataSettings: TransformationSettings,
    eventEmitter: EventEmitter,
): PipelineModel<ModelRepresentation> {
    const transformations = transformationsFactory(dataSettings.transformations);
    const preScaler = normalizeFunctionFactory(dataSettings.normalization);
    const postScaler = transformations.length
        ? normalizeFunctionFactory(dataSettings.normalization)
        : undefined;

    const featureTransform = {
        preScaler,
        postScaler,
        transformations,
    };

    return new PipelineModel(model, featureTransform, eventEmitter);
}
