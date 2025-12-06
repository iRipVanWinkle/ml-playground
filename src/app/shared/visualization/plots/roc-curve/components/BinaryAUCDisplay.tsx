import type { BinaryRocCurveData } from '../types';
import { AUCCard } from './AUCCard';

type BinaryAUCDisplayProps = {
    rocCurveData: BinaryRocCurveData;
};

export function BinaryAUCDisplay({ rocCurveData }: BinaryAUCDisplayProps) {
    return (
        <div className="flex flex-col">
            <AUCCard
                label="AUC (Area Under Curve)"
                value={rocCurveData.auc}
                tooltip="Measures how well the model can distinguish between positive and negative classes. Higher is better, with 1.0 being perfect."
                className="text-lg"
            />
        </div>
    );
}
