import { useEffect, useState } from 'react';
import { Button, Collapsible } from '@/app/shared/ui';
import { ChevronDown, RotateCcw } from 'lucide-react';
import { cn } from '@/app/shared/ui/utils';
import type { Dataset } from '@/app/shared/types';
import { useModel } from '../hooks/useModel';
import { Inputs } from './Inputs';
import { useModelDefinition } from '@/app/models/ui-registry';
import {
    resetUserExample as reset,
    setUserExampleInputs as setInputs,
    useTrainingReport,
    useUserExample,
} from '@/app/store';

type UserExampleProps = {
    dataset: Dataset;
};

export function UserExample({ dataset }: UserExampleProps) {
    const { id, headers, categories, isImage } = dataset;

    if (id === null || isImage) {
        return null;
    }

    return <UserExampleContent datasetId={id} headers={headers} categories={categories} />;
}

type UserExampleContentProps = {
    datasetId: string;
    headers: string[];
    categories?: string[];
};

function UserExampleContent({ datasetId, headers, categories }: UserExampleContentProps) {
    const [isOpen, setIsOpen] = useState(false);

    useEffect(() => reset(), [datasetId]);

    const report = useTrainingReport();
    const { inputs, result } = useUserExample();
    const { prediction, probabilities } = result ?? {};

    const { runPrediction } = useModel({ datasetId });
    const { visualization } = useModelDefinition(report.type);
    const PredictionComponent = visualization.predictionComponent;

    const [target, ...features] = headers;

    useEffect(() => {
        if (inputs) {
            runPrediction(inputs, report);
        }
    }, [report, inputs, runPrediction]);

    return (
        <Collapsible open={isOpen} onOpenChange={setIsOpen} className="w-full">
            <div className="flex items-center justify-between">
                <Collapsible.Trigger className="flex items-center gap-2 py-3 text-left">
                    <span className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                        Test Example
                    </span>
                    <ChevronDown
                        className={cn(
                            'h-4 w-4 text-muted-foreground transition-transform duration-200',
                            isOpen && 'rotate-180',
                        )}
                    />
                </Collapsible.Trigger>
                {isOpen && (
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={reset}
                        className="text-xs text-muted-foreground"
                    >
                        <RotateCcw className="h-3 w-3" />
                        Reset
                    </Button>
                )}
            </div>

            <Collapsible.Content className="flex flex-col gap-3">
                <Inputs features={features} inputs={inputs} onChange={setInputs} />

                {PredictionComponent &&
                    (prediction !== undefined ? (
                        <PredictionComponent
                            taskType={report.taskType}
                            target={target}
                            categories={categories}
                            prediction={prediction}
                            probabilities={probabilities}
                        />
                    ) : (
                        <div className="rounded-lg bg-muted/30 p-4 flex flex-col gap-3">
                            <div className="text-center text-xs font-medium uppercase tracking-wider text-muted-foreground">
                                Prediction
                            </div>
                            <div className="flex items-center justify-center py-4 text-muted-foreground">
                                <span className="text-sm">
                                    {inputs
                                        ? 'Run training to see prediction'
                                        : 'Enter values to see prediction'}
                                </span>
                            </div>
                        </div>
                    ))}
            </Collapsible.Content>
        </Collapsible>
    );
}
