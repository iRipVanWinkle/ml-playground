import { LayoutGrid, NotebookPen } from 'lucide-react';
import { AVAILABLE_DESIGNS_LIST } from './constants';
import { useDesign } from './useDesign';
import type { DesignVariant } from './types';
import { Button } from '@/app/shared/ui';

const DESIGN_LABELS: Record<DesignVariant, string> = {
    classic: 'Classic',
    notebook: 'Notebook',
};

export function DesignSwitch() {
    const { design, setDesign } = useDesign();

    if (AVAILABLE_DESIGNS_LIST.length <= 1) {
        return null;
    }

    const currentIndex = AVAILABLE_DESIGNS_LIST.indexOf(design);
    const nextDesign = AVAILABLE_DESIGNS_LIST[(currentIndex + 1) % AVAILABLE_DESIGNS_LIST.length];
    const label = `Switch to ${DESIGN_LABELS[nextDesign]} design`;

    return (
        <Button
            variant="ghost"
            size="icon"
            onClick={() => setDesign(nextDesign)}
            aria-label={label}
            title={label}
        >
            {design === 'classic' ? (
                <LayoutGrid className="size-4" />
            ) : (
                <NotebookPen className="size-4" />
            )}
        </Button>
    );
}
