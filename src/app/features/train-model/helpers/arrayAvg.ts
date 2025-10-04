export function arrayAvg(arr: number[]): number {
    if (arr.length === 0) {
        return 0;
    }
    const sum = arr.reduce((acc, val) => acc + val, 0);
    return sum / arr.length;
}
