import { Card, CardContent, CardHeader, CardTitle } from '@/app/components/ui/card';
import { Input } from '@/app/components/ui/input';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectSeparator,
    SelectTrigger,
    SelectValue,
} from '@/app/components/ui/select';
import { Slider } from '@/app/components/ui/slider';
import { Label } from '@/app/components/ui/label';
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createFileFromURL } from '@/app/lib/utils';
import { Switch } from '@/app/components/ui/switch';
import Transformations from './Transformations';
import {
    extractFeatures,
    resetTrainingReport,
    setNormalizationFunction,
    setTransformation,
    useDataSettings,
    useIsTraining,
    useNumTrainInputFeatures,
    useTaskType,
    type DataSettings,
    type NormalizationFunction,
} from '@/app/store';
import { InfoTooltip } from '@/app/components/ui/info-tooltip';

type DataSectionState = {
    file: File | null;
    datasetPath?: string;
    shuffleData: boolean;
    trainTestSplit: number;
};

type TransformationArray = DataSettings['transformations'];

const DEFAULT_STATE: DataSectionState = {
    file: null,
    shuffleData: true,
    trainTestSplit: 80,
};

const PREPERED_REGRESSION_DATASETS = [
    {
        value: './data/world-happiness-report-2017 1(in).csv',
        label: 'World happiness report 2017 (Happiness.Score, Economy..GDP.per.Capita.)',
    },
    {
        value: './data/world-happiness-report-2017 2(in).csv',
        label: 'World happiness report 2017 (Happiness.Score, Economy..GDP.per.Capita., Freedom)',
    },
    {
        value: './data/non-linear-regression-x-y.csv',
        label: 'Non linear regression',
    },
    {
        value: './data/linear-relationship.csv',
        label: 'Linear relationship',
    },
    {
        value: './data/quadratic-relationship.csv',
        label: 'Quadratic relationship',
    },
    {
        value: './data/wave-pattern-regression.csv',
        label: 'Wave pattern regression',
    },
];

const PREPERED_CLASSIFICATION_DATASETS = [
    {
        value: './data/mnist-number-0-1.csv',
        label: 'MNIST numbers (0, 1)',
    },
    {
        value: './data/microchips-tests.csv',
        label: 'Microchips Tests (non linear)',
    },
];

export default function DataSection() {
    const taskType = useTaskType();
    const dataSettings = useDataSettings();
    const numFeatures = useNumTrainInputFeatures();
    const isTraining = useIsTraining();

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
        async function prepereData() {
            await extractFeatures({
                file: state.file!,
                shuffleData: state.shuffleData,
                trainTestSplit: state.trainTestSplit,
                taskType: taskTypeRef.current,
            });
        }

        if (state.file) {
            resetTrainingReport();
            prepereData();
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

    const handleChangeNormalization = (value: string) => {
        setNormalizationFunction(value as NormalizationFunction);
    };

    const handleTransformation = (transformations: TransformationArray) => {
        setTransformation(transformations);
    };

    const datasets =
        taskType === 'regression' ? PREPERED_REGRESSION_DATASETS : PREPERED_CLASSIFICATION_DATASETS;

    return (
        <Card className="gap-5">
            <CardHeader>
                <CardTitle>Dataset</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-5">
                <div className="grid gap-2">
                    <Label>
                        Dataset{' '}
                        <InfoTooltip>
                            Choose your dataset from available options or upload a new one.
                        </InfoTooltip>
                    </Label>
                    <Select
                        value={state.datasetPath ?? (state.file ? 'custom' : '')}
                        onValueChange={handleChangeDataset}
                        disabled={isTraining}
                    >
                        <SelectTrigger className="w-full truncate">
                            <SelectValue placeholder="Select dataset" />
                        </SelectTrigger>
                        <SelectContent>
                            {datasets.map((dataset) => (
                                <SelectItem key={dataset.value} value={dataset.value}>
                                    {dataset.label}
                                </SelectItem>
                            ))}
                            <SelectSeparator />
                            <SelectItem value="custom">Custom Dataset</SelectItem>
                        </SelectContent>
                    </Select>
                    {state.datasetPath === 'custom' && (
                        <Input
                            type="file"
                            accept=".csv"
                            disabled={isTraining}
                            onChange={(e) =>
                                handleChange({ file: e.target.files ? e.target.files[0] : null })
                            }
                        />
                    )}
                </div>
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
                            disabled={isTraining}
                            checked={state.shuffleData}
                            onCheckedChange={(checked) =>
                                handleChange({ shuffleData: checked === true })
                            }
                        />
                    </Label>
                </div>
                <div className="grid gap-2">
                    <Label>Train/Test Split</Label>
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
                        disabled={isTraining}
                        onValueChange={(value) => handleChange({ trainTestSplit: value[0] })}
                    />
                </div>

                <div className="grid gap-2">
                    <Label>
                        Normalization{' '}
                        <InfoTooltip>
                            Scale features to improve model training.{' '}
                            <a href="#" className="text-blue-400 hover:underline">
                                Show more
                            </a>
                        </InfoTooltip>
                    </Label>
                    <Select
                        disabled={isTraining}
                        value={dataSettings.normalization}
                        onValueChange={handleChangeNormalization}
                    >
                        <SelectTrigger className="w-50">
                            <SelectValue placeholder="Select normalization" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="none">None</SelectItem>
                            <SelectItem value="zscore">Z-Score</SelectItem>
                        </SelectContent>
                    </Select>
                </div>
                <div className="grid gap-2">
                    <Label>
                        Transformations{' '}
                        <InfoTooltip>
                            Apply feature transformations features to improve model performance.
                        </InfoTooltip>
                    </Label>
                    <Transformations
                        transformations={dataSettings.transformations}
                        numFeatures={numFeatures}
                        disabled={isTraining}
                        onChange={(value) => handleTransformation(value as TransformationArray)}
                    />
                </div>
            </CardContent>
        </Card>
    );
}
