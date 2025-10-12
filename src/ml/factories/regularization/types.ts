export type RegularizationType = 'none' | 'l1' | 'l2' | 'elasticnet';

type RegularizationNoneConfig = {
    type: 'none';
};

type RegularizationLConfig = {
    type: 'l1' | 'l2';
    lambda: number;
};

type RegularizationElasticNetConfig = {
    type: 'elasticnet';
    lambda: number;
    alpha: number;
};

export type RegularizationConfig =
    | RegularizationNoneConfig
    | RegularizationLConfig
    | RegularizationElasticNetConfig;
