import { memory } from '@tensorflow/tfjs';
import { workerLogUtils } from './logs';

const DEFAULT_CONSECUTIVE_THRESHOLD = 5;

/**
 * Detects potential memory leaks by tracking consecutive increases in tensor count.
 */
export class MemoryLeakDetector {
    private previousTensorCount: number | null = null;
    private consecutiveTensorIncreases = 0;
    private threshold: number;

    constructor(threshold: number = DEFAULT_CONSECUTIVE_THRESHOLD) {
        this.threshold = threshold;
        this.initialize();
    }

    /**
     * Initializes or resets the detector with current memory state.
     */
    initialize(): void {
        this.previousTensorCount = memory().numTensors;
        this.consecutiveTensorIncreases = 0;
    }

    /**
     * Checks current tensor memory and logs a warning if leak threshold is exceeded.
     * Call this during each training iteration in development mode.
     */
    check(): void {
        if (!import.meta.env.DEV) return;

        const currentMemory = memory();
        const currentTensorCount = currentMemory.numTensors;

        if (this.previousTensorCount !== null) {
            if (currentTensorCount > this.previousTensorCount) {
                this.consecutiveTensorIncreases++;

                if (this.consecutiveTensorIncreases >= this.threshold) {
                    workerLogUtils.logMemoryWarning(
                        this.consecutiveTensorIncreases,
                        this.previousTensorCount,
                        currentTensorCount,
                    );
                }
            } else {
                this.consecutiveTensorIncreases = 0;
            }
        }

        this.previousTensorCount = currentTensorCount;
    }

    /**
     * Resets the detector state.
     */
    reset(): void {
        this.previousTensorCount = null;
        this.consecutiveTensorIncreases = 0;
    }
}
