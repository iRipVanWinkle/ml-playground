import { createElement } from 'react';
import { AlertTriangle, Shapes, Share2, TrendingUp } from 'lucide-react';
import type { TaskType } from '@/app/shared/types';

type TaskTypeObject = {
    value: TaskType;
    label: string;
    description: string;
    icon: React.ReactElement;
};

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
    {
        value: 'anomaly',
        label: 'Anomaly',
        description: 'Identify unusual patterns or outliers in data',
        icon: createElement(AlertTriangle),
    },
] as TaskTypeObject[];
