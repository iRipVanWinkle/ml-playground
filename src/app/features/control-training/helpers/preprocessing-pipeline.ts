import type { Model, ModelRepresentation } from '@/ml/types';
import { PipelineModel } from '@/ml/models';
import { EventEmitter } from '@/ml/events/EventEmitter';
import type { TransformationSettings } from '@/app/shared/types';
import { normalizeFunctionFactory, transformationsFactory } from '@/ml/factories';

export function createPreprocessingPipeline<T extends ModelRepresentation>(
    model: Model<T>,
    dataSettings: TransformationSettings,
    eventEmitter?: EventEmitter,
): PipelineModel<T> {
    const preperedTransformations = dataSettings.transformations.filter((t) => t.type !== '');
    const transformations = transformationsFactory(preperedTransformations);
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
