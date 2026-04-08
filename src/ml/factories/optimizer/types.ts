type OptimizerBasicConfig = {
    maxIterations: number;
    tolerance?: number;
    learningRate: number;
    scheduler: boolean;
    schedulerConfig: { s0: number | undefined; p: number | undefined };
};

type OptimizerBatchConfig = OptimizerBasicConfig & {
    type: 'batch';
};

type OptimizerSGDConfig = OptimizerBasicConfig & {
    type: 'sgd';
    batchSize: number;
};

type OptimizerMomentumConfig = OptimizerBasicConfig & {
    type: 'momentum';
    beta: number;
};

type OptimizerAdaConfig = OptimizerBasicConfig & {
    type: 'adam';
    beta1: number;
    beta2: number;
};

export type OptimizerConfig =
    | OptimizerBatchConfig
    | OptimizerSGDConfig
    | OptimizerMomentumConfig
    | OptimizerAdaConfig;
