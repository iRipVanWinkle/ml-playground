import { Select } from '@/app/shared/ui';
interface TreeSelectorProps {
    value: number;
    onChange: (value: number) => void;
    amount: number;
}

export function TreeSelector({ value, onChange, amount }: TreeSelectorProps) {
    return (
        <Select value={String(value)} onValueChange={(val) => onChange(Number(val))}>
            <Select.Trigger id="selectedTree" size="xs" variant="transparent">
                <Select.Value placeholder="Select a tree" />
            </Select.Trigger>
            <Select.Content>
                {Array.from({ length: amount }, (_, index) => (
                    <Select.Item key={index} value={String(index)}>
                        Tree {index + 1}
                    </Select.Item>
                ))}
            </Select.Content>
        </Select>
    );
}
