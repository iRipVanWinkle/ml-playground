import { useMemo } from 'react';

interface UseMatrixLabelsParams {
    labels: string[];
    selectedView: string;
    isBinaryClassification: boolean;
}

interface UseMatrixLabelsReturn {
    rowLabels: string[];
    columnLabels: string[];
    classLabels: string[];
}

/**
 * Hook to transform labels for matrix display based on classification type and view mode
 */
export function useMatrixLabels({
    labels,
    selectedView,
    isBinaryClassification,
}: UseMatrixLabelsParams): UseMatrixLabelsReturn {
    const isOneVsRest = selectedView !== 'full';
    const targetClassIndex = isOneVsRest ? labels.findIndex((label) => label === selectedView) : -1;

    return useMemo<UseMatrixLabelsReturn>(() => {
        if (isBinaryClassification) {
            const [positiveClass, negativeClass] = labels;
            return {
                rowLabels: [`${positiveClass} (+ve)`, `${negativeClass} (-ve)`],
                columnLabels: [`${positiveClass} (+ve)`, `${negativeClass} (-ve)`],
                classLabels: labels,
            };
        }

        if (isOneVsRest) {
            const baseLabels = [labels[targetClassIndex], 'Rest'];
            return {
                rowLabels: baseLabels,
                columnLabels: baseLabels,
                classLabels: [labels[targetClassIndex], 'Rest'],
            };
        }

        return {
            rowLabels: labels,
            columnLabels: labels,
            classLabels: labels,
        };
    }, [isBinaryClassification, isOneVsRest, labels, targetClassIndex]);
}
