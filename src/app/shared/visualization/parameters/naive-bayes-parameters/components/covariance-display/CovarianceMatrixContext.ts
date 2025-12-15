import { createContext } from 'react';
import { EMPTY_MATRIX_LIKE, type MatrixLike } from '@/app/shared/helpers';

export const CovarianceMatrixContext = createContext<{
    covariances: MatrixLike;
    featureLabels: string[];
}>({
    covariances: EMPTY_MATRIX_LIKE,
    featureLabels: [],
});
