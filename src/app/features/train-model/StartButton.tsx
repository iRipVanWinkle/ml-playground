import { SplitButton, type MenuItemsType } from '@/app/shared/ui';
import { Play, StepForward } from 'lucide-react';
import { useState } from 'react';

type StartMode = 'full' | 'step';

type StartButtonProps = {
    onTrain: (byStep?: boolean) => void;
    disabled?: boolean;
};

export function StartButton({ onTrain, disabled }: StartButtonProps) {
    const [startMode, setStartMode] = useState<StartMode>('full');

    const handleTrain = (byStep = false) => {
        setStartMode(byStep ? 'step' : 'full');
        onTrain(byStep);
    };

    const menuItems = [
        {
            key: 'full',
            label: (
                <>
                    <Play /> Start Training
                </>
            ),
            onSelect: () => setStartMode('full'),
        },
        {
            key: 'step',
            label: (
                <>
                    <StepForward /> Start Step-by-Step
                </>
            ),
            onSelect: () => setStartMode('step'),
        },
    ] as Array<MenuItemsType & { key: StartMode }>;

    const currentMode = menuItems.find((it) => it.key === (startMode ?? 'full'));

    return (
        <SplitButton
            onClick={() => handleTrain(startMode === 'step')}
            disabled={disabled}
            data-testid="start-training"
            menuItems={menuItems}
        >
            {currentMode?.label}
        </SplitButton>
    );
}
