import { createContext } from 'react';

interface DecisionTreeContextType {
    featureLabels?: string[];
    categories?: string[];
}

export const DecisionTreeContext = createContext<DecisionTreeContextType>({});
