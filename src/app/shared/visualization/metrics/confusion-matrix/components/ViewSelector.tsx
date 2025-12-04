import { Select } from '@/app/shared/ui';

interface ViewSelectorProps {
    value: string;
    onChange: (value: string) => void;
    labels: string[];
}

export function ViewSelector({ value, onChange, labels }: ViewSelectorProps) {
    return (
        <Select value={value} onValueChange={onChange}>
            <Select.Trigger id="selectedClass" size="xs" variant="transparent">
                <Select.Value placeholder="Select a class" />
            </Select.Trigger>
            <Select.Content>
                <Select.Item value="full">Full Matrix</Select.Item>
                <Select.Group>
                    <Select.Label>One vs Rest</Select.Label>
                    {labels.map((label, index) => (
                        <Select.Item key={`class-${index}`} value={label}>
                            {label}
                        </Select.Item>
                    ))}
                </Select.Group>
            </Select.Content>
        </Select>
    );
}
