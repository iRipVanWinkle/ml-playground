export const DEFAULT_OPTIMIZER = {
    type: 'batch',
    maxIterations: 100,
    tolerance: 0.0001,
    learningRate: 0.01,
    scheduler: false,
    schedulerConfig: { s0: 1, p: 0.5 },
} as const;
