export type ThetaInitializationType =
    | 'zeros'
    | 'ones'
    | 'constant'
    | 'uniform'
    | 'normal'
    | 'xavierUniform'
    | 'xavierNormal'
    | 'heUniform'
    | 'heNormal';

type ThetaInitializationBaseConfig = {
    type: 'zeros' | 'ones' | 'xavierUniform' | 'xavierNormal' | 'heUniform' | 'heNormal';
};

type ThetaInitializationConstantConfig = {
    type: 'constant';
    value: number;
};

type ThetaInitializationUniformConfig = {
    type: 'uniform';
    min: number;
    max: number;
};

type ThetaInitializationNormalConfig = {
    type: 'normal';
    mean: number;
    stddev: number;
};

export type ThetaInitializationConfig =
    | ThetaInitializationBaseConfig
    | ThetaInitializationConstantConfig
    | ThetaInitializationUniformConfig
    | ThetaInitializationNormalConfig;
