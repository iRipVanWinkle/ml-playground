/**
 * Logging utilities for UI components in development mode.
 */
export const uiLogUtils = {
    /**
     * Logs metrics received event with count and iteration info.
     * @param count - The number of metrics received.
     * @param iteration - The current iteration.
     */
    logMetricsReceived(count: number, iteration: number | string): void {
        if (import.meta.env.DEV) {
            console.log(
                `%c[Hook] %cReceived metrics #${count} (iteration: ${iteration})`,
                'color: #2196f3; font-weight: bold',
                'color: #4caf50',
            );
        }
    },

    /**
     * Logs training completion with total metrics count.
     * @param count - Total number of metrics received.
     */
    logTrainingComplete(count: number): void {
        if (import.meta.env.DEV) {
            console.log(
                `%c[Hook] %cTraining completed - Total metrics received: ${count}`,
                'color: #2196f3; font-weight: bold',
                'color: #4caf50; font-weight: bold',
            );
        }
    },
};
