import {
    constantInitializer,
    heNormalInitializer,
    heUniformInitializer,
    normalInitializer,
    onesInitializer,
    uniformInitializer,
    xavierNormalInitializer,
    xavierUniformInitializer,
    zerosInitializer,
    type ThetaInitializer,
} from '@/ml/utils/theta';
import type { ThetaInitializationConfig } from '../store';

export function getThetaInitializer(
    thetaInitialization: ThetaInitializationConfig,
): ThetaInitializer {
    switch (thetaInitialization.type) {
        case 'ones':
            return onesInitializer();
        case 'constant':
            return constantInitializer(thetaInitialization.value);
        case 'uniform':
            return uniformInitializer(thetaInitialization.min, thetaInitialization.max);
        case 'normal':
            return normalInitializer(thetaInitialization.mean, thetaInitialization.stddev);
        case 'xavierUniform':
            return xavierUniformInitializer();
        case 'xavierNormal':
            return xavierNormalInitializer();
        case 'heUniform':
            return heUniformInitializer();
        case 'heNormal':
            return heNormalInitializer();
        case 'zeros':
        default:
            return zerosInitializer();
    }
}
