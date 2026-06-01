import { Card, Separator } from '@/app/shared/ui';
import {
    useDataset,
    useHasData,
    useModelSettings,
    useTrainingState,
    useTransformations,
} from '@/app/store';
import { Controls } from '@/app/features/control-training';
import {
    ModelDataPlot,
    ParametersVisualization,
    TabbedVisualizations,
    TrainingMetricsGrid,
    TrainingProgress,
} from '@/app/features/visualize-training';
import { UserExample } from '@/app/features/user-example';

export function TrainingSection() {
    const state = useTrainingState();
    const hasData = useHasData();
    const modelSettings = useModelSettings();
    const anyTransformations = useTransformations();
    const dataset = useDataset();

    const modelType = modelSettings.type;

    const transformations = anyTransformations.filter((t) => t.type !== '');
    const isIdle = state === 'idle';
    const isInit = state === 'init';

    return (
        <Card key={modelType}>
            <Card.Content className="flex flex-col gap-4">
                <TrainingProgress
                    modelType={modelType}
                    modelSettings={modelSettings}
                    dataset={dataset}
                    controlsComponent={<Controls hasData={hasData} />}
                />

                {isIdle ? (
                    <p className="text-sm text-muted-foreground">
                        Pick a dataset to get started - once it's loaded, you'll be ready to train
                        your model.
                    </p>
                ) : (
                    <>
                        {!isInit && <TrainingMetricsGrid modelType={modelType} />}

                        <div className="flex flex-col gap-4">
                            <ModelDataPlot
                                modelType={modelType}
                                dataset={dataset}
                                modelSettings={modelSettings}
                            />

                            {isInit ? (
                                <>
                                    <Separator />
                                    <p className="text-sm text-muted-foreground">
                                        Your data is ready! Hit <strong>Start Training</strong> to
                                        watch your model learn and unlock detailed metrics and
                                        visualizations.
                                    </p>
                                </>
                            ) : (
                                <>
                                    <UserExample dataset={dataset} />

                                    <Separator />

                                    <TabbedVisualizations modelType={modelType} dataset={dataset} />

                                    <ParametersVisualization
                                        modelType={modelType}
                                        modelSettings={modelSettings}
                                        transformations={transformations}
                                        dataset={dataset}
                                    />
                                </>
                            )}
                        </div>
                    </>
                )}
            </Card.Content>
        </Card>
    );
}
