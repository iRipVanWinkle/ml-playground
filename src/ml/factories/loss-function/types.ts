export type LossFunctionType =
    | 'mse'
    | 'mae'
    | 'huber'
    | 'logcosh'
    | 'binaryCrossentropy'
    | 'categoricalCrossentropy'
    | 'logitsBasedBinaryCrossentropy'
    | 'logitsBasedCategoricalCrossentropy';

type LossFunctionGeneralConfig = {
    type: Exclude<LossFunctionType, 'huber'>;
};

type LossFunctionHuberConfig = {
    type: 'huber';
    delta: number;
};

export type LossFunctionConfig = LossFunctionGeneralConfig | LossFunctionHuberConfig;
