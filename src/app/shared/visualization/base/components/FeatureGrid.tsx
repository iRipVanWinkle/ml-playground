import { Collapsible } from '@/app/shared/ui';
import { useState, type ComponentType, type CSSProperties } from 'react';
import { ChevronDown } from 'lucide-react';

const INITIAL_VISIBLE_COUNT = 10;

type FeatureGridProps = {
    items: number[];
    labels: string[];
    visibleCount?: number;
    oneColumn?: boolean;
    itemComponent: ComponentType<{ label: string; value: number; maxAbs: number }>;
};

function FeatureGridRoot({
    items,
    labels,
    itemComponent,
    oneColumn = false,
    visibleCount = INITIAL_VISIBLE_COUNT,
}: FeatureGridProps) {
    const [isOpen, setIsOpen] = useState(false);

    const visibleIndices = Math.min(visibleCount, items.length);
    const visibleItems = items.slice(0, visibleIndices);
    const hiddenItems = items.slice(visibleIndices);
    const hasMore = hiddenItems.length > 0;

    let maxAbsItem = 0.001;
    for (let i = 0; i < items.length; i++) {
        const absValue = Math.abs(items[i]);
        if (absValue > maxAbsItem) {
            maxAbsItem = absValue;
        }
    }

    const Item = itemComponent;

    return (
        <Collapsible open={isOpen} onOpenChange={setIsOpen}>
            <div
                className={`grid grid-cols-1 gap-x-8 gap-y-3 ${!oneColumn ? 'md:grid-cols-2' : ''} `}
            >
                {visibleItems.map((item, index) => (
                    <Item
                        key={index}
                        label={labels[index] || `Feature ${index + 1}`}
                        value={item}
                        maxAbs={maxAbsItem}
                    />
                ))}
            </div>
            {hasMore && (
                <>
                    <Collapsible.Content>
                        <div
                            className={`mt-2 grid grid-cols-1 gap-2 gap-x-8 ${!oneColumn ? 'md:grid-cols-2' : ''}`}
                        >
                            {hiddenItems.map((item, index) => {
                                const actualIndex = index + visibleIndices;
                                return (
                                    <Item
                                        key={actualIndex}
                                        label={labels[actualIndex] || `Feature ${actualIndex + 1}`}
                                        value={item}
                                        maxAbs={maxAbsItem}
                                    />
                                );
                            })}
                        </div>
                    </Collapsible.Content>
                    <Collapsible.Trigger className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors mt-2">
                        <ChevronDown
                            className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                        />
                        {isOpen ? 'Show less' : `Show ${hiddenItems.length} more...`}
                    </Collapsible.Trigger>
                </>
            )}
        </Collapsible>
    );
}

type FeatureGridRowProps = {
    label: string;
    children: number;
    progressStyle: CSSProperties;
    withSign?: boolean;
};

export function FeatureGridRow({
    label,
    children,
    progressStyle,
    withSign = false,
}: FeatureGridRowProps) {
    const usedPrecision = getAdaptivePrecision(children);
    const formattedValue =
        withSign && children > 0
            ? `+${children.toFixed(usedPrecision)}`
            : children.toFixed(usedPrecision);

    return (
        <div className="flex items-center px-2 pb-3 border-b border-border/50">
            <div
                className={`flex-1 truncate text-left text-sm text-muted-foreground`}
                title={label}
            >
                {label}
            </div>
            <div className="ml-3 flex shrink-0 items-center gap-2">
                <div className="h-2 w-12 overflow-hidden rounded-full bg-muted">
                    <div className="h-full rounded-full transition-all" style={progressStyle} />
                </div>
                <div className={`w-20 text-right text-sm font-medium tabular-nums`}>
                    {formattedValue}
                </div>
            </div>
        </div>
    );
}

function getAdaptivePrecision(value: number): number {
    const absValue = Math.abs(value);
    if (absValue >= 1000000) return 0;
    if (absValue >= 100000) return 1;
    if (absValue >= 10000) return 2;
    if (absValue >= 1000) return 3;
    return 4;
}

const FeatureGrid = Object.assign(FeatureGridRoot, {
    Cell: FeatureGridRow,
});

export { FeatureGrid };
