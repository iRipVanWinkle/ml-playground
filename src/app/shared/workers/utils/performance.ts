/**
 * Performance measurement utilities for development.
 */
export const performanceUtils = {
    /**
     * Logs the latency of a worker message in development mode.
     * @param label - The label for the message (e.g., "[Worker -> Client]").
     * @param type - The message type.
     * @param sentAt - The timestamp when the message was sent.
     * @param color - The color for the label.
     */
    logLatency(label: string, type: string, sentAt?: number, color = '#00bcd4'): void {
        if (import.meta.env.DEV && sentAt) {
            const now = performance.now() + performance.timeOrigin;
            const latency = now - sentAt;
            console.log(
                `%c${label} %c${type} %clatency: ${latency.toFixed(2)}ms`,
                `color: ${color}; font-weight: bold`,
                'color: inherit',
                'color: #4caf50',
            );
        }
    },

    /**
     * Gets the current timestamp for performance measurement.
     * @returns The current timestamp in development mode, or undefined otherwise.
     */
    getTimestamp(): number | undefined {
        return import.meta.env.DEV ? performance.now() + performance.timeOrigin : undefined;
    },

    /**
     * Logs the duration of an operation in development mode.
     * @param label - The label for the operation.
     * @param operation - The operation name.
     * @param startTime - The timestamp when the operation started.
     * @param color - The color for the label.
     */
    logDuration(label: string, operation: string, startTime: number, color = '#9c27b0'): void {
        if (import.meta.env.DEV) {
            const duration = performance.now() - startTime;
            console.log(
                `%c${label} %c${operation} %cduration: ${duration.toFixed(2)}ms`,
                `color: ${color}; font-weight: bold`,
                'color: inherit',
                'color: #4caf50',
            );
        }
    },
};
