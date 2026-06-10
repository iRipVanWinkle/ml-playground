import { createContext } from 'react';
import type { DesignVariant } from './types';

export type DesignContextValue = {
    design: DesignVariant;
    setDesign: (design: DesignVariant) => void;
};

export const DesignContext = createContext<DesignContextValue | null>(null);
