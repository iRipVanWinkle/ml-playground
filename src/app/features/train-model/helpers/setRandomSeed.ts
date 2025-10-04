import { Randomizer } from '@/ml/random/Randomizer';

export function setRandomSeed(seed?: number): void {
    Randomizer.setSeed(seed);
}
