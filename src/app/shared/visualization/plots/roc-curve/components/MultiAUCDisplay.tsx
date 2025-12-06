import type { MulticlassRocCurveData } from '../types';
import { AUCCard } from './AUCCard';

type MultiAUCDisplayProps = {
    rocCurveData: MulticlassRocCurveData;
    categories: string[];
};

type AUCConfig = {
    label: string;
    value: 'macroAuc' | 'weightedAuc';
    tooltip: string;
};

const AUC_CONFIG: AUCConfig[] = [
    {
        label: 'Macro AUC',
        value: 'macroAuc',
        tooltip:
            'Average of how well the model separates each class from others. Each class counts equally.',
    },
    {
        label: 'Weighted AUC',
        value: 'weightedAuc',
        tooltip:
            'Average of how well the model separates each class from others. Larger classes have more influence.',
    },
];

export function MultiAUCDisplay({ rocCurveData, categories }: MultiAUCDisplayProps) {
    return (
        <>
            <div className="grid grid-cols-2 gap-3">
                {AUC_CONFIG.map((config) => (
                    <div className="p-4 rounded-lg bg-primary-foreground">
                        <AUCCard
                            label={config.label}
                            value={rocCurveData[config.value]}
                            tooltip={config.tooltip}
                            className="text-base"
                        />
                    </div>
                ))}
            </div>
            <div className="text-sm font-semibold text-foreground">AUC per Class</div>
            <div className="p-4 rounded-lg bg-primary-foreground">
                <div className="grid grid-cols-3 md:grid-cols-5 lg:grid-cols-5 gap-3 text-sm">
                    {rocCurveData.curves.map((_, index) => {
                        const classIndex = rocCurveData.classIndices[index];
                        const label = categories?.[classIndex] || `Class ${classIndex}`;
                        const auc = rocCurveData.aucs[index];
                        return <AUCCard key={classIndex} label={label} value={auc} />;
                    })}
                </div>
            </div>
        </>
    );
}
