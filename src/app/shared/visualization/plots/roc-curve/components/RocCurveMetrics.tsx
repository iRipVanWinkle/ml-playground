import type { RocCurveData } from '../types';

type RocCurveMetricsProps = {
    rocCurveData: RocCurveData;
    categories: string[];
};

export function RocCurveMetrics({ rocCurveData, categories }: RocCurveMetricsProps) {
    return (
        <div className="p-4 rounded-lg bg-muted flex flex-col gap-3">
            {rocCurveData.type === 'binary' ? (
                <div className="flex flex-col">
                    <div className="text-center">
                        <div className="text-muted-foreground text-xs font-medium mb-1">
                            AUC (Area Under Curve)
                        </div>
                        <div className="text-lg font-semibold">{rocCurveData.auc.toFixed(3)}</div>
                    </div>
                </div>
            ) : (
                <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-lg text-center bg-primary-foreground">
                        <div className="text-muted-foreground text-xs font-medium mb-1">
                            Macro AUC
                        </div>
                        <div className="text-base font-semibold">
                            {rocCurveData.macroAuc.toFixed(3)}
                        </div>
                    </div>
                    <div className="p-3 rounded-lg text-center bg-primary-foreground">
                        <div className="text-muted-foreground text-xs font-medium mb-1">
                            Weighted AUC
                        </div>
                        <div className="text-base font-semibold">
                            {rocCurveData.weightedAuc.toFixed(3)}
                        </div>
                    </div>
                </div>
            )}
            {rocCurveData.type === 'multiclass' && (
                <>
                    <div className="text-sm font-semibold text-foreground">AUC per Class</div>
                    <div className="p-3 rounded-lg bg-primary-foreground">
                        <div className="grid grid-cols-3 md:grid-cols-5 lg:grid-cols-5 gap-3 text-sm">
                            {rocCurveData.curves.map((_, index) => {
                                const classIndex = rocCurveData.classIndices[index];
                                const label = categories?.[classIndex] || `Class ${classIndex}`;
                                const auc = rocCurveData.aucs[index];
                                return (
                                    <div key={classIndex} className="text-center">
                                        <div className="text-muted-foreground text-xs">{label}</div>
                                        <div className="font-semibold">{auc.toFixed(3)}</div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
