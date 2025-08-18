function comb(n: number, k: number): number {
    if (k > n) return 0;
    k = Math.min(k, n - k); // Use symmetry
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

export const calculateOutputFeatures = (
    type: string,
    degree: number,
    numFeatures: number,
): number => {
    switch (type) {
        case 'sinusoid':
            return numFeatures * degree;
        case 'polynomial':
            // Polynomial features are generated from all combinations of features raised to powers up to degree
            return countNewPolynomialFeatures(numFeatures, degree);
        default:
            return 0; // Unknown type generates no features
    }
};
