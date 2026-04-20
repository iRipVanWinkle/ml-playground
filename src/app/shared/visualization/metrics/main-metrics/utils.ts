export function isNumber(value: number | null | undefined): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}
