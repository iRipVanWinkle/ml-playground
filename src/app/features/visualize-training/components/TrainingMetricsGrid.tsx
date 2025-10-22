import type { ModelType } from '@/app/models/types';
import { useModelDefinition } from '@/app/models/ui-registry';
import { useTrainingReport } from '../store';

type TrainingMetricsGridProps = {
    modelType: ModelType;
};

export function TrainingMetricsGrid({ modelType }: TrainingMetricsGridProps) {
    const report = useTrainingReport();
    const modelDefinition = useModelDefinition(modelType);

    const MetricsGrid = modelDefinition.visualization.metricsGridComponent;

    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <MetricsGrid report={report} />
        </div>
    );
}
