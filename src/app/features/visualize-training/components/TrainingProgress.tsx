import { Progress } from '@/app/shared/ui';
import type { ModelSettings } from '@/app/models/types';
import type { ModelType } from '@/app/models/types';
import { useModelDefinition } from '@/app/models/ui-registry';
import { useTrainingReport } from '../store';
import type { Dataset } from '@/app/shared/types';

type TrainingProgressProps = {
    controlsComponent: React.ReactNode;
    modelType: ModelType;
    modelSettings: ModelSettings;
    dataset: Dataset;
};

export function TrainingProgress({
    controlsComponent,
    modelType,
    modelSettings,
    dataset,
}: TrainingProgressProps) {
    const report = useTrainingReport();
    const modelDefinition = useModelDefinition(modelType);

    const progressInfo = modelDefinition.progress.getProgressInfo({
        report,
        dataset,
        settings: modelSettings,
    });

    return (
        <div className="grid gap-4 sticky top-0 z-10 bg-card/80 backdrop-blur-md pt-4 -mt-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="flex gap-4">{controlsComponent}</div>

                <div
                    className="flex items-center justify-end"
                    data-testid="training-progress"
                    aria-live="polite"
                >
                    {progressInfo.label}
                </div>
            </div>

            <Progress
                value={progressInfo.type === 'determinate' ? progressInfo.current : undefined}
                max={progressInfo.type === 'determinate' ? progressInfo.max : undefined}
                className="w-full bg-gray-200 rounded-full h-0.25"
            />
        </div>
    );
}
