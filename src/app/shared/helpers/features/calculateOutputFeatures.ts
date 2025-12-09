export const calculateOutputFeatures = (
    type: string,
    degree: number,
    numFeatures: number,
): number => {
    switch (type) {
        case 'sinusoid':
        case 'cosinusoid':
            return numFeatures * degree;
        case 'fourier':
            return numFeatures * degree * 2;
        case 'polynomial':
            return countNewPolynomialFeatures(numFeatures, degree);
        default:
            return 0; // Unknown type generates no features
    }
};

export const calculateOutputFeatureLabels = (
    type: string,
    degree: number,
    featureLabels: string[],
): string[] => {
    switch (type) {
        case 'sinusoid':
            return generateTrigLabels('sin', degree, featureLabels);
        case 'cosinusoid':
            return generateTrigLabels('cos', degree, featureLabels);
        case 'fourier':
            return [
                ...generateTrigLabels('sin', degree, featureLabels),
                ...generateTrigLabels('cos', degree, featureLabels),
            ];
        case 'polynomial':
            return generatePolynomialCombinations(featureLabels.length, degree).map((exp) =>
                formatPolynomialLabel(exp, featureLabels),
            );
        default:
            return []; // Unknown type generates no features
    }
};

function comb(n: number, k: number): number {
    if (k > n) return 0;
    k = Math.min(k, n - k);
    let res = 1;
    for (let i = 1; i <= k; i++) {
        res *= n - i + 1;
        res /= i;
    }
    return res;
}

function countNewPolynomialFeatures(numFeatures: number, degree: number): number {
    if (degree < 2) return 0; // no new features if degree < 2

    let total = 0;
    for (let d = 2; d <= degree; d++) {
        total += comb(numFeatures + d - 1, d);
    }
    return total;
}

function generatePolynomialCombinations(numFeatures: number, degree: number): number[][] {
    if (degree < 2) return [];

    const results: number[][] = [];
    const combo: number[] = Array(numFeatures).fill(0);

    function recurse(pos: number, remaining: number) {
        if (pos === numFeatures - 1) {
            combo[pos] = remaining;
            results.push([...combo]);
            return;
        }
        for (let i = 0; i <= remaining; i++) {
            combo[pos] = i;
            recurse(pos + 1, remaining - i);
        }
    }

    for (let d = 2; d <= degree; d++) {
        recurse(0, d);
    }

    return results;
}

function formatPolynomialLabel(exponents: number[], featureLabels: string[]): string {
    return exponents
        .map((exp, i) =>
            exp > 0 ? (exp === 1 ? featureLabels[i] : `${featureLabels[i]}^${exp}`) : '',
        )
        .filter(Boolean)
        .join('*');
}

function generateTrigLabels(
    func: 'sin' | 'cos',
    degree: number,
    featureLabels: string[],
): string[] {
    return [...Array(degree).keys()].flatMap((d) =>
        featureLabels.map((f) => (d === 0 ? `${func}(${f})` : `${func}(${d + 1}*${f})`)),
    );
}
