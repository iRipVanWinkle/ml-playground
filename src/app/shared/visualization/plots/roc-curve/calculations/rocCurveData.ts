import type { MatrixLike } from '@/app/shared/helpers';
import { auc, macroAverage, multiclassRocCurve, rocCurve, weightedAverage } from '@/ml/metrics';
import type { RocCurveData } from '../types';

export function rocCurveData(
    yTrue: MatrixLike,
    yProb: MatrixLike,
    confusionMatrix: number[][],
): RocCurveData {
    const isBinaryClassification = yProb.shape[1] === 1;

    if (isBinaryClassification) {
        const binaryCurve = rocCurve(yTrue.array, yProb.array);
        const aucScore = auc(binaryCurve.fpr, binaryCurve.tpr);

        return {
            type: 'binary',
            auc: aucScore,
            ...binaryCurve,
        };
    }

    const multiclassCurves = multiclassRocCurve(yTrue, yProb);
    const aucs = multiclassCurves.curves.map((classCurve) => auc(classCurve.fpr, classCurve.tpr));

    return {
        type: 'multiclass',
        aucs,
        macroAuc: macroAverage(aucs),
        weightedAuc: weightedAverage(aucs, confusionMatrix),
        ...multiclassCurves,
    };
}
