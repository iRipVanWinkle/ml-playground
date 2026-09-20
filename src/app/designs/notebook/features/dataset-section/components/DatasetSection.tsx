import { useState } from 'react';
import { InlineSelect, Section, StepNum } from '../../../shared';
import { setDataset, useDataset, useIsTraining, useRandomSeed, useTaskType } from '@/app/store';
import { getDatasetsForTask } from '@/app/features/load-dataset';
import { createFileFromURL } from '@/app/features/load-dataset/libs/file-fetcher';
import { extractFeatures } from '@/app/features/load-dataset/libs/extract-features';

export function DatasetSection() {
    const taskType = useTaskType();
    const dataset = useDataset();
    const randomSeed = useRandomSeed();
    const isTraining = useIsTraining();
    const [isLoading, setIsLoading] = useState(false);

    const availableDatasets = getDatasetsForTask(taskType);
    const selectedDataset = availableDatasets.find((d) => d.value === dataset.id);

    const handleDatasetChange = async (value: string) => {
        const option = availableDatasets.find((d) => d.value === value);
        if (!option) return;

        setIsLoading(true);
        try {
            const file = await createFileFromURL(value, 'dataset.csv');
            const data = await extractFeatures({
                file,
                shuffleData: true,
                trainTestSplit: 80,
                taskType,
                seed: randomSeed,
            });

            setDataset({
                ...data,
                id: value,
                isImage: option.isImage,
            });
        } catch (error) {
            console.error('Failed to load dataset:', error);
        } finally {
            setIsLoading(false);
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
            </Section.Body>
        </Section>
    );
}
