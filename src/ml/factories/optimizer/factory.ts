import type { EventEmitter } from '../../events/EventEmitter';
import { AdamGD, BatchGD, MomentumGD, StochasticGD } from '../../optimizers';
import type { TrainingControl } from '../../types';
import { LearningRate } from '../../LearningRate';
import type { OptimizerConfig } from './types';

export function learningRateFactory(
    rate: number,
    schedulerConfig?: { s0?: number; p?: number },
): LearningRate | number {
    let learningRate: LearningRate | number = rate;
    if (schedulerConfig) {
        const { s0, p } = schedulerConfig;
        learningRate = new LearningRate(learningRate, s0, p);
    }

    return learningRate;
}

export function optimizerFactory(
    optimizerConfig: OptimizerConfig,
    eventEmitter: EventEmitter,
    trainingController: TrainingControl,
) {
    const { scheduler, schedulerConfig, maxIterations, tolerance } = optimizerConfig;

    const learningRate = learningRateFactory(
        optimizerConfig.learningRate,
        scheduler ? schedulerConfig : undefined,
    );

    const baseConfig = {
        learningRate,
        maxIterations,
        tolerance,
        eventEmitter,
        trainingController,
    };

    switch (optimizerConfig.type) {
        case 'adam': {
            const { beta1, beta2 } = optimizerConfig;
            return new AdamGD({ ...baseConfig, beta1, beta2 });
        }

        case 'momentum': {
            const { beta } = optimizerConfig;
            return new MomentumGD({ ...baseConfig, beta });
        }

        case 'sgd': {
            const { batchSize } = optimizerConfig;
            return new StochasticGD({ ...baseConfig, batchSize });
        }

        case 'batch':
            return new BatchGD(baseConfig);
    }
}
