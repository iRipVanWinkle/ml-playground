import { createElement } from 'react';
import { Shapes, Share2, TrendingUp } from 'lucide-react';

export const TASK_TYPES = [
    {
        value: 'regression',
        label: 'Regression',
        description: 'Predict continuous values like prices, temperatures, or sales',
        icon: createElement(TrendingUp),
    },
    {
        value: 'classification',
        label: 'Classification',
        description: 'Categorize data into distinct classes or groups',
        icon: createElement(Shapes),
    },
    {
        value: 'clustering',
        label: 'Clustering',
        description: 'Group similar data points together without predefined labels',
        icon: createElement(Share2),
    },
];
