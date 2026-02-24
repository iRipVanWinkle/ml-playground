import { Select } from '@/app/shared/ui';

interface TrainTestSelectorProps {
    value: string;
    onChange: (value: string) => void;
}

export function TrainTestSelector({ value, onChange }: TrainTestSelectorProps) {
    return (
        <Select value={value} onValueChange={onChange}>
            <Select.Trigger
                id="datasetSplit"
                size="xs"
                variant="transparent"
                aria-label="Select dataset split"
            >
                <Select.Value placeholder="Select a dataset split" />
            </Select.Trigger>
            <Select.Content>
                <Select.Item value="train">Train</Select.Item>
                <Select.Item value="test">Test</Select.Item>
            </Select.Content>
        </Select>
    );
}
