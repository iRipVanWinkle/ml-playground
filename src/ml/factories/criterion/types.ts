export type CriterionType = 'mse' | 'mae' | 'huber' | 'logcosh' | 'gini' | 'entropy';

type CriterionGeneralConfig = {
    type: Exclude<CriterionType, 'huber'>;
};

type CriterionHuberConfig = {
    type: 'huber';
    delta: number;
};

export type CriterionConfig = CriterionGeneralConfig | CriterionHuberConfig;
