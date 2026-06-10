import type { DesignVariant } from './types';

export const DESIGN_STORAGE_KEY = 'design';

export const DEFAULT_DESIGN: DesignVariant = 'classic';

const NOTEBOOK_DESIGN_PARAM = 'design';

function isNotebookEnabled(): boolean {
    if (typeof window === 'undefined') return false;

    return new URLSearchParams(window.location.search).get(NOTEBOOK_DESIGN_PARAM) === '1';
}

export function getAvailableDesigns(): DesignVariant[] {
    return isNotebookEnabled() ? ['classic', 'notebook'] : ['classic'];
}

export const AVAILABLE_DESIGNS_LIST: DesignVariant[] = getAvailableDesigns();
