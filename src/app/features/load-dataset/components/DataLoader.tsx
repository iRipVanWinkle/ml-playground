import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { resetTrainingReport, useTaskType } from '@/app/store';
import {
    DEFAULT_STATE,
    PREPARED_CLASSIFICATION_DATASETS,
    PREPARED_REGRESSION_DATASETS,
} from '../constants/datasets';
import type { DataSectionProps, DataSectionState } from '../store/types';
import { createFileFromURL } from '../libs/file-fetcher';
import { Field, Input, Label, Select, Slider, Switch } from '@/app/shared/ui';
import { extractFeatures } from '../store/actions';

export function DataLoader({ disabled }: DataSectionProps) {
    const taskType = useTaskType();

    const [state, setState] = useState<DataSectionState>(DEFAULT_STATE);

    const taskTypeRef = useRef(taskType);
    useLayoutEffect(() => {
        taskTypeRef.current = taskType;
        setState((prev) => ({
            ...prev,
            file: null,
            datasetPath: '',
        }));
    }, [taskType]);

    useEffect(() => {
        if (state.file) {
            resetTrainingReport();
            extractFeatures({
                file: state.file!,
                shuffleData: state.shuffleData,
                trainTestSplit: state.trainTestSplit,
                taskType: taskTypeRef.current,
            });
        }
    }, [state]);

    const handleChange = (data: Partial<DataSectionState>) => {
        setState((prev) => ({ ...prev, ...data }));
    };

    const handleChangeDataset = async (value: string) => {
        if (value === 'custom') {
            setState((prev) => ({ ...prev, file: null, datasetPath: 'custom' }));
        } else {
            const file = await createFileFromURL(value, 'dataset.csv');
            setState((prev) => ({ ...prev, file, datasetPath: value }));
        }
    };

    const datasets =
        taskType === 'regression' ? PREPARED_REGRESSION_DATASETS : PREPARED_CLASSIFICATION_DATASETS;

    return (
        <>
            <Field label="Dataset" htmlFor="datasetSelect">
                <Select
                    value={state.datasetPath ?? (state.file ? 'custom' : '')}
                    onValueChange={handleChangeDataset}
                    disabled={disabled}
                >
                    <Select.Trigger
                        id="datasetSelect"
                        className="w-full truncate"
                        data-testid="dataset-select"
                    >
                        <Select.Value placeholder="Select dataset" />
                    </Select.Trigger>
                    <Select.Content>
                        {datasets.map((dataset) => (
                            <Select.Item key={dataset.value} value={dataset.value}>
                                {dataset.label}
                            </Select.Item>
                        ))}
                        <Select.Separator />
                        <Select.Item value="custom" data-testid="custom-dataset-option">
                            Custom Dataset
                        </Select.Item>
                    </Select.Content>
                </Select>
                {state.datasetPath === 'custom' && (
                    <Input
                        data-testid="custom-dataset-input"
                        type="file"
                        accept=".csv"
                        disabled={disabled}
                        onChange={(e) =>
                            handleChange({ file: e.target.files ? e.target.files[0] : null })
                        }
                    />
                )}
            </Field>
            <div className="flex items-center gap-2">
                <Label
                    htmlFor="shuffle"
                    className="w-full hover:bg-accent/50 flex items-center justify-between gap-3 rounded-lg border p-3 transition-colors has-[[aria-checked=false]]:text-muted-foreground"
                >
                    <div className="grid gap-1.5 font-normal text-left">
                        <p className="text-sm leading-none font-medium transition-colors">
                            Shuffle Data
                        </p>
                        <p className="text-muted-foreground text-xs">
                            Randomly shuffle the dataset before splitting to ensure unbiased
                            training
                        </p>
                    </div>
                    <Switch
                        id="shuffle"
                        data-testid="shuffle-switch"
                        disabled={disabled}
                        checked={state.shuffleData}
                        onCheckedChange={(checked) =>
                            handleChange({ shuffleData: checked === true })
                        }
                    />
                </Label>
            </div>
            <Field label="Train/Test Split">
                <div className="flex justify-between">
                    <span className="text-xs text-muted-foreground">
                        Train ({state.trainTestSplit}%)
                    </span>
                    <span className="text-xs text-muted-foreground">
                        Test ({100 - state.trainTestSplit}%)
                    </span>
                </div>
                <Slider
                    defaultValue={[state.trainTestSplit]}
                    max={100}
                    min={1}
                    step={1}
                    disabled={disabled}
                    onValueChange={(value) => handleChange({ trainTestSplit: value[0] })}
                />
            </Field>
        </>
    );
}
