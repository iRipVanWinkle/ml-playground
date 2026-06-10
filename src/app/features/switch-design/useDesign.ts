import { useContext } from 'react';
import { DesignContext } from './context';

export function useDesign() {
    const context = useContext(DesignContext);

    if (!context) {
        throw new Error('useDesign must be used within a DesignProvider');
    }

    return context;
}
