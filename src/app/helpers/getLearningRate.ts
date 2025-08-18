import { LearningRate } from '@/ml/LearningRate';

export function getLearningRate(
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
