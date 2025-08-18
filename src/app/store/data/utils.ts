export function extractFeaturesAndLabels(data: (string | number)[][]): {
    features: number[][];
    labels: number[][];
} {
    return {
        features: data.map((row) => row.slice(1).map(Number)),
        labels: data.map((row) => [Number(row[0])]),
    };
}

export function calculateMinMax(data: number[][]): { xMin: number[]; xMax: number[] } {
    return data.reduce(
        (acc, row) => {
            row.forEach((value, index) => {
                acc.xMin[index] = Math.min(value, acc.xMin[index]);
                acc.xMax[index] = Math.max(value, acc.xMax[index]);
            });
            return acc;
        },
        { xMin: Array(data[0].length).fill(Infinity), xMax: Array(data[0].length).fill(-Infinity) },
    );
}

export function generateCartesianProduct(
    predictionsNum: number,
    xMin: number[],
    xMax: number[],
): number[][] {
    // Generate predefined number of values for each axis between corresponding min and max values.
    const axes = xMin.map((min, index) =>
        Array.from(
            { length: predictionsNum },
            (_, i) => min + (i * (xMax[index] - min)) / (predictionsNum - 1),
        ),
    );

    // Compute Cartesian product dynamically for any number of columns
    const cartesianProduct = (arrays: number[][]): number[][] => {
        return arrays.reduce<number[][]>(
            (acc, curr) => acc.flatMap((prev) => curr.map((value) => [...prev, value])),
            [[]],
        );
    };

    return cartesianProduct(axes);
}
