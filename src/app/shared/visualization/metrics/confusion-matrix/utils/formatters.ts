/**
 * Formats a value as a percentage with 1 decimal place
 */
export function formatPercentage(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
}

/**
 * Formats a number with specified decimal places (default 3)
 */
export function formatDecimal(value: number, decimals: number = 3): string {
    return value.toFixed(decimals);
}

/**
 * Calculates the percentage of a value within a row total
 */
export function getPercentage(value: number, rowTotal: number): number {
    if (rowTotal === 0) return 0;
    return Math.round((value / rowTotal) * 100);
}
