import { Select } from '@/app/shared/ui';
interface ViewSelectorProps {
    value: string;
    onChange: (value: string) => void;
}

export function ViewSelector({ value, onChange }: ViewSelectorProps) {
    return (
        <Select value={value} onValueChange={onChange}>
            <Select.Trigger id="selectedView" size="xs" variant="transparent">
                <Select.Value placeholder="Select a view" />
            </Select.Trigger>
            <Select.Content>
                <Select.Item value="graph">Graph View</Select.Item>
                <Select.Item value="raw">Raw View</Select.Item>
            </Select.Content>
        </Select>
    );
}
