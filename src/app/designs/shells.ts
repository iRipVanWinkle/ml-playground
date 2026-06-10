import { lazy, type ComponentType } from 'react';
import type { DesignVariant } from '@/app/features/switch-design';

export const DESIGN_SHELLS: Record<DesignVariant, ComponentType> = {
    classic: lazy(() => import('./classic')),
    notebook: lazy(() => import('./notebook')),
};
