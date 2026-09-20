import { useState } from 'react';
import { InlineSelect, InlineSelectInput, Section, StepNum } from '../../../shared';
import {
    setDataset,
    setRandomSeed,
    useDataset,
    useIsTraining,
    useRandomSeed,
    useTaskType,
} from '@/app/store';
import { getDatasetsForTask } from '@/app/features/load-dataset';
import { createFileFromURL } from '@/app/features/load-dataset/libs/file-fetcher';
import { extractFeatures } from '@/app/features/load-dataset/libs/extract-features';

export function DatasetSection() {
    const taskType = useTaskType();
    const dataset = useDataset();
    const randomSeed = useRandomSeed();
    const isTraining = useIsTraining();
    const [isLoading, setIsLoading] = useState(false);
    const [trainTestSplit, setTrainTestSplit] = useState(80);
    const [shuffleData, setShuffleData] = useState(true);

    const availableDatasets = getDatasetsForTask(taskType);
    const selectedDataset = availableDatasets.find((d) => d.value === dataset.id);

    const loadDataset = async (
        datasetValue: string,
        split = trainTestSplit,
        seed = randomSeed,
        shuffle = shuffleData,
    ) => {
        const option = availableDatasets.find((d) => d.value === datasetValue);
        if (!option) return;

        setIsLoading(true);
        try {
            const file = await createFileFromURL(datasetValue, 'dataset.csv');
            const data = await extractFeatures({
                file,
                shuffleData: shuffle,
                trainTestSplit: split,
                taskType,
                seed,
            });

            setDataset({
                ...data,
                id: datasetValue,
                isImage: option.isImage,
            });
        } catch (error) {
            console.error('Failed to load dataset:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleDatasetChange = async (value: string) => {
        await loadDataset(value);
    };

    const handleShuffleChange = (value: string | number | boolean) => {
        const nextShuffle = Boolean(value);
        setShuffleData(nextShuffle);
        if (selectedDataset) {
            void loadDataset(selectedDataset.value, trainTestSplit, randomSeed, nextShuffle);
        }
    };

    const handleSeedChange = (value: string | number | boolean) => {
        const nextSeed = value === '' || value === undefined ? undefined : Number(value);
        setRandomSeed(nextSeed);
        if (selectedDataset) {
            void loadDataset(selectedDataset.value, trainTestSplit, nextSeed, shuffleData);
        }
    };

    const handleSplitChange = (value: string | number | boolean) => {
        const nextSplit = Number(value);
        setTrainTestSplit(nextSplit);
        if (selectedDataset) {
            void loadDataset(selectedDataset.value, nextSplit, randomSeed, shuffleData);
        }
    };

    return (
        <Section step={2} total={7}>
            <Section.Header>
                <StepNum />
                <Section.Title>The dataset</Section.Title>
            </Section.Header>
            <Section.Body>
                <p className="text-2xl font-light">
                    Learning from{' '}
                    <InlineSelect
                        value={selectedDataset?.value ?? ''}
                        onValueChange={handleDatasetChange}
                        disabled={isTraining || isLoading}
                    >
                        <InlineSelect.Trigger placeholder="pick a dataset" />
                        <InlineSelect.Content>
                            {availableDatasets.map((ds) => (
                                <InlineSelect.Item key={ds.value} value={ds.value} hint={ds.hint}>
                                    {ds.label}
                                </InlineSelect.Item>
                            ))}
                        </InlineSelect.Content>
                    </InlineSelect>
                    {selectedDataset?.hint && ` — ${selectedDataset.hint}`}
                </p>
                <p>
                    Every model learns from examples. Here we look at the raw rows and columns
                    before any transformation happens.
                </p>
                <p>
                    Data is{' '}
                    <InlineSelectInput
                        value={shuffleData}
                        onValueChange={handleShuffleChange}
                        disabled={isTraining || isLoading}
                    >
                        <InlineSelectInput.Trigger placeholder="shuffle">
                            {shuffleData ? 'shuffled' : 'unshuffled'}
                        </InlineSelectInput.Trigger>
                        <InlineSelectInput.Content>
                            <InlineSelectInput.Label>Shuffle</InlineSelectInput.Label>
                            <InlineSelectInput.Toggle onLabel="Shuffled" offLabel="Unshuffled" />
                            <InlineSelectInput.Hint>
                                Randomize row order before split
                            </InlineSelectInput.Hint>
                            <InlineSelectInput.Footer>
                                <InlineSelectInput.Done>Done</InlineSelectInput.Done>
                            </InlineSelectInput.Footer>
                        </InlineSelectInput.Content>
                    </InlineSelectInput>{' '}
                    with seed{' '}
                    <InlineSelectInput
                        value={randomSeed ?? ''}
                        onValueChange={handleSeedChange}
                        disabled={isTraining || isLoading}
                    >
                        <InlineSelectInput.Trigger placeholder="none">
                            {randomSeed}
                        </InlineSelectInput.Trigger>
                        <InlineSelectInput.Content>
                            <InlineSelectInput.Label>Random seed</InlineSelectInput.Label>
                            <InlineSelectInput.Input type="number" min={0} max={9999} />
                            <InlineSelectInput.Hint>
                                Same seed → same shuffle → reproducible
                            </InlineSelectInput.Hint>
                            <InlineSelectInput.Footer>
                                <InlineSelectInput.Done>Done</InlineSelectInput.Done>
                            </InlineSelectInput.Footer>
                        </InlineSelectInput.Content>
                    </InlineSelectInput>
                    , then split{' '}
                    <InlineSelectInput
                        value={trainTestSplit}
                        onValueChange={handleSplitChange}
                        disabled={isTraining || isLoading}
                    >
                        <InlineSelectInput.Trigger placeholder="split">
                            {trainTestSplit}/{100 - trainTestSplit}
                        </InlineSelectInput.Trigger>
                        <InlineSelectInput.Content>
                            <InlineSelectInput.Label>
                                Train / test split —{' '}
                                <b className="font-bold normal-case text-foreground">
                                    {trainTestSplit}/{100 - trainTestSplit}
                                </b>
                            </InlineSelectInput.Label>
                            <InlineSelectInput.Range min={50} max={95} />
                            <InlineSelectInput.Hint>
                                Percent used for training; the rest is held out
                            </InlineSelectInput.Hint>
                            <InlineSelectInput.Footer>
                                <InlineSelectInput.Done>Done</InlineSelectInput.Done>
                            </InlineSelectInput.Footer>
                        </InlineSelectInput.Content>
                    </InlineSelectInput>{' '}
                    between training and held-out test. Training runs on WebGPU — everything stays
                    in your browser, no data leaves this tab.
                </p>
            </Section.Body>
        </Section>
    );
}
