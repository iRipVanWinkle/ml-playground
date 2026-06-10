import { useEffect, useState, type ReactNode } from 'react';
import { AVAILABLE_DESIGNS_LIST, DEFAULT_DESIGN, DESIGN_STORAGE_KEY } from './constants';
import { DesignContext } from './context';
import type { DesignVariant } from './types';

function readStoredDesign(): DesignVariant {
    if (typeof window === 'undefined') return DEFAULT_DESIGN;

    const stored = window.localStorage.getItem(DESIGN_STORAGE_KEY) as DesignVariant | null;

    return stored && AVAILABLE_DESIGNS_LIST.includes(stored) ? stored : DEFAULT_DESIGN;
}

export function DesignProvider({ children }: { children: ReactNode }) {
    const [design, setDesignState] = useState<DesignVariant>(readStoredDesign);

    useEffect(() => {
        if (typeof window !== 'undefined') {
            window.localStorage.setItem(DESIGN_STORAGE_KEY, design);
        }
    }, [design]);

    const setDesign = (next: DesignVariant) => setDesignState(next);

    return (
        <DesignContext.Provider value={{ design, setDesign }}>{children}</DesignContext.Provider>
    );
}
