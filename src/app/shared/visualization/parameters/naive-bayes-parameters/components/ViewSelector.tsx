import { Select } from '@/app/shared/ui';
interface ViewSelectorProps {
    value: string;
    onChange: (value: string) => void;
    classLabels: string[];
}

export function ViewSelector({ value, onChange, classLabels }: ViewSelectorProps) {
    return (
        <Select value={value} onValueChange={onChange}>
            <Select.Trigger
                id="selectedClass"
                size="xs"
                variant="transparent"
                aria-label="Select class"
            >
                <Select.Value placeholder="Select class" />
            </Select.Trigger>
            <Select.Content>
                <Select.Item value="all">All Classes</Select.Item>
                {classLabels?.map((label, index) => (
                    <Select.Item key={`class-${index}`} value={String(index)}>
                        {label}
                    </Select.Item>
                ))}
                <Select.Item value="raw">Raw Parameters</Select.Item>
            </Select.Content>
        </Select>
    );
}
