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

export function labelEncoding(data: (string | number)[][], targetClass?: string): string[] {
    const firstColumn = data.map((row) => row[0]);
    const originalLabels = Array.from(new Set(firstColumn)).map((label) => label.toString());

    const isBinary = originalLabels.length === 2;

    // Reorder originalLabels directly based on the target class
    if (isBinary) {
        const [label1, label2] = originalLabels;
        const hasZeroAndOne = originalLabels.includes('0') && originalLabels.includes('1');

        if (targetClass && targetClass === label1) {
            originalLabels[0] = label1;
            originalLabels[1] = label2;
        } else if (targetClass && targetClass === label2) {
            originalLabels[0] = label2;
            originalLabels[1] = label1;
        } else if (hasZeroAndOne) {
            originalLabels[0] = '1';
            originalLabels[1] = '0';
        }
    }

    // Create labelMap from ordered labels (index = encoding value)
    const labelMap: Record<string, number> = {};
    originalLabels.forEach((label, index) => {
        labelMap[label] = index;
    });

    data.forEach((row) => {
        row[0] = labelMap[row[0].toString()]; // Convert labels to numeric
    });
    return originalLabels;
}
