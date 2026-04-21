import { memory } from '@tensorflow/tfjs';

/**
 * Logging utilities for web workers in development mode.
 */
export const workerLogUtils = {
    /**
     * Logs an informational message from a worker.
     * @param message - The message to log.
     * @param data - Optional additional data to log.
     */
    logInfo(message: string, data?: unknown): void {
        if (import.meta.env.DEV) {
            console.log(
                `%c[Worker] %c${message}`,
                'color: #9c27b0; font-weight: bold',
                'color: inherit',
            );
            if (data !== undefined) {
                console.log(data);
            }
        }
    },

    /**
     * Logs an error message from a worker.
     * @param message - The error message to log.
     * @param error - Optional error object.
     */
    logError(message: string, error?: unknown): void {
        console.error(`[Worker] ${message}`, error ?? '');
    },

    /**
     * Logs a warning message from a worker.
     * @param message - The warning message to log.
     * @param data - Optional additional data to log.
     */
    logWarning(message: string, data?: unknown): void {
        if (import.meta.env.DEV) {
            console.warn(
                `%c[Worker] %c${message}`,
                'color: #ff9800; font-weight: bold',
                'color: inherit',
            );
            if (data !== undefined) {
                console.warn(data);
            }
        }
    },

    /**
     * Logs a memory leak warning with tensor count information.
     * @param consecutiveIncreases - Number of consecutive increases.
     * @param previousCount - Previous tensor count.
     * @param currentCount - Current tensor count.
     * @param memoryInfo - Current memory information.
     */
    logMemoryWarning(
        consecutiveIncreases: number,
        previousCount: number,
        currentCount: number,
    ): void {
        if (import.meta.env.DEV) {
            const memoryInfo = memory();
            console.warn(
                `%c[Worker] %cMemory Leak Warning: %cTensor count has increased for ${consecutiveIncreases} consecutive iterations (${previousCount} → ${currentCount})`,
                'color: #ff9800; font-weight: bold',
                'color: #f44336; font-weight: bold',
                'color: inherit',
            );
            console.warn('Current memory:', memoryInfo);
        }
    },

    /**
     * Logs training lifecycle events (started/finished) with memory info.
     * @param event - The event type ('started' or 'finished').
     * @param memoryInfo - Current memory information.
     */
    logTrainingLifecycle(event: 'started' | 'finished'): void {
        if (import.meta.env.DEV) {
            console.info(`Training ${event}`, memory());
        }
    },
};
