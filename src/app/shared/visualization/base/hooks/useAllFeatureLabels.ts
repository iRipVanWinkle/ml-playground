import { useMemo } from 'react';
import type { Transformation } from '@/app/shared/types';
import { calculateOutputFeatureLabels } from '@/app/shared/helpers/features/calculateOutputFeatures';

export function useAllFeatureLabels(headers: string[], transformations: Transformation[]) {
    return useMemo(() => {
        const originalLabels = headers.slice(1); // except first column (target)
        const transformedLabels = transformations.flatMap((transformation) =>
            calculateOutputFeatureLabels(
                transformation.type,
                transformation.degree,
                originalLabels,
            ),
        );

        return [...originalLabels, ...transformedLabels];
    }, [headers, transformations]);
}
