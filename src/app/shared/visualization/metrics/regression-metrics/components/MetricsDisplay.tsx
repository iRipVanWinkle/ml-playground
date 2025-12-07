import type { RegressionMetricsData } from '../types';
import { MetricCard } from './MetricCard';

interface MetricsDisplayProps {
    metrics: RegressionMetricsData;
}

interface MetricConfig {
    label: string;
    field: keyof RegressionMetricsData;
    tooltip: string;
}

const REGRESSION_METRICS_CONFIG: MetricConfig[] = [
    {
        label: 'MSE',
        field: 'mse',
        tooltip:
            'Measures how far predictions are from real values. Larger errors are punished more.',
    },
    {
        label: 'RMSE',
        field: 'rmse',
        tooltip:
            'Square root of MSE. Shows error in the same units as the target, making it easier to read.',
    },
    {
        label: 'MAE',
        field: 'mae',
        tooltip:
            'Measures average distance between predictions and real values. Treats all errors equally.',
    },
    {
        label: 'R² Score',
        field: 'r2',
        tooltip:
            'Shows how much variation in the target is explained by the model. Range: (-∞, 1], where 1 is perfect prediction.',
    },
];

export function MetricsDisplay({ metrics }: MetricsDisplayProps) {
    return (
        <div className="p-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {REGRESSION_METRICS_CONFIG.map((metricConfig) => (
                    <MetricCard
                        key={metricConfig.label}
                        label={metricConfig.label}
                        value={metrics[metricConfig.field]}
                        tooltip={metricConfig.tooltip}
                    />
                ))}
            </div>
        </div>
    );
}
