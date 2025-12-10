import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { Collapsible } from '@/app/shared/ui';
import { BiasTerm } from './BiasTerm';
import { WeightItem } from './WeightItem';

const INITIAL_VISIBLE_COUNT = 10;

type ParametersDisplayProps = {
    bias: number;
    weights: number[];
    categoryName?: string;
    featureLabels: string[];
    precision?: number;
};

export function ParametersDisplay({
    bias,
    weights,
    categoryName,
    featureLabels,
    precision = 4,
}: ParametersDisplayProps) {
    const [isOpen, setIsOpen] = useState(false);

    const visibleWeights = weights.slice(0, INITIAL_VISIBLE_COUNT);
    const hiddenWeights = weights.slice(INITIAL_VISIBLE_COUNT);
    const hasMoreWeights = hiddenWeights.length > 0;
    const maxAbsWeight = Math.max(...weights.map(Math.abs), 0.001);
    const hasHeader = categoryName && categoryName.length > 0;

    return (
        <div className="rounded-lg bg-primary-foreground p-4 flex flex-col gap-3">
            {hasHeader && <h4 className="text-base font-semibold text-primary">{categoryName}</h4>}

            <BiasTerm bias={bias} precision={precision} />

            <Collapsible open={isOpen} onOpenChange={setIsOpen}>
                <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                    {visibleWeights.map((weight, index) => (
                        <WeightItem
                            key={index}
                            featureName={featureLabels[index] || `Feature ${index + 1}`}
                            weight={weight}
                            maxAbsWeight={maxAbsWeight}
                            precision={precision}
                        />
                    ))}
                </div>
                {hasMoreWeights && (
                    <>
                        <Collapsible.Content>
                            <div className="mt-2 grid grid-cols-1 gap-2 md:grid-cols-2">
                                {hiddenWeights.map((weight, index) => {
                                    const actualIndex = index + INITIAL_VISIBLE_COUNT;
                                    return (
                                        <WeightItem
                                            key={actualIndex}
                                            featureName={
                                                featureLabels[actualIndex] ||
                                                `Feature ${actualIndex + 1}`
                                            }
                                            weight={weight}
                                            maxAbsWeight={maxAbsWeight}
                                            precision={precision}
                                        />
                                    );
                                })}
                            </div>
                        </Collapsible.Content>
                        <Collapsible.Trigger className="mt-3 flex items-center gap-1 text-sm font-medium text-primary hover:underline">
                            <ChevronDown
                                className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                            />
                            {isOpen ? 'Show less' : `Show ${hiddenWeights.length} more...`}
                        </Collapsible.Trigger>
                    </>
                )}
            </Collapsible>
        </div>
    );
}
