import { Select } from '@/app/shared/ui';
interface ViewSelectorProps {
    value: string;
    onChange: (value: string) => void;
    classLabels: string[];
}

export function ViewSelector({ value, onChange, classLabels }: ViewSelectorProps) {
    const isBinary = classLabels.length <= 2;

    return (
        <Select value={value} onValueChange={onChange}>
            <Select.Trigger id="selectedClass" size="xs" variant="transparent">
                <Select.Value placeholder="Select a class" />
            </Select.Trigger>
            <Select.Content>
                <Select.Item value="all">
                    {isBinary ? 'Formatted Parameters' : 'All Classes'}
                </Select.Item>
                {!isBinary &&
                    classLabels.map((label, index) => (
                        <Select.Item key={`class-${index}`} value={String(index)}>
                            {label}
                        </Select.Item>
                    ))}
                <Select.Item value="raw">Raw Parameters</Select.Item>
            </Select.Content>
        </Select>
    );
}
