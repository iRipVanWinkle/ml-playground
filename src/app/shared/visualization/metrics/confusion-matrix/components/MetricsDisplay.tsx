import type {
    BinaryConfusionMatrixMetrics,
    ConfusionMatrixMetrics,
    MulticlassConfusionMatrixMetrics,
} from '../types';
import { MetricCard } from './MetricCard';

interface MetricsDisplayProps {
    metrics: ConfusionMatrixMetrics;
}

interface PerClassMetricConfig {
    label: string;
    field: keyof Omit<BinaryConfusionMatrixMetrics, 'type'> | null;
    tooltip: string;
    format: 'percentage' | 'decimal';
}

interface FullMetricConfig {
    label: string;
    field: keyof Omit<MulticlassConfusionMatrixMetrics, 'type'> | null;
    tooltip: string;
    format: 'percentage' | 'decimal';
}

const PER_CLASS_METRICS_CONFIG: PerClassMetricConfig[] = [
    {
        label: 'Accuracy',
        field: 'accuracy',
        tooltip: 'Proportion of correct predictions out of all predictions.',
        format: 'percentage',
    },
    {
        label: 'MCC',
        field: 'mcc',
        tooltip:
            'Measures how well predictions match actual values. Ranges from -1 to 1, where 1 is perfect.',
        format: 'decimal',
    },
    {
        label: "Cohen's Kappa",
        field: 'cohensKappa',
        tooltip:
            'Measures how well predictions match actual values, considering that some agreement might happen by luck.',
        format: 'decimal',
    },
    {
        label: 'Precision',
        field: 'precision',
        tooltip: 'Proportion of positive predictions that are correct.',
        format: 'percentage',
    },
    {
        label: 'Recall',
        field: 'recall',
        tooltip: 'Proportion of actual positives that were correctly identified.',
        format: 'percentage',
    },
    {
        label: 'F1',
        field: 'f1',
        tooltip:
            'Combines precision and recall into a single score. Balances both metrics equally.',
        format: 'percentage',
    },
];

const FULL_METRICS_CONFIG: FullMetricConfig[] = [
    {
        label: 'Accuracy',
        field: 'accuracy',
        tooltip: 'Proportion of correct predictions out of all predictions.',
        format: 'percentage',
    },
    {
        label: 'MCC',
        field: 'mcc',
        tooltip:
            'Measures how well predictions match actual values. Ranges from -1 to 1, where 1 is perfect.',
        format: 'decimal',
    },
    {
        label: "Cohen's Kappa",
        field: 'cohensKappa',
        tooltip:
            'Measures how well predictions match actual values, considering that some agreement might happen by luck.',
        format: 'decimal',
    },
    {
        label: 'Macro Precision',
        field: 'macroPrecision',
        tooltip:
            'Average precision across all classes. Each class counts equally, regardless of size.',
        format: 'percentage',
    },
    {
        label: 'Macro Recall',
        field: 'macroRecall',
        tooltip: 'Average of recall across all classes.',
        format: 'percentage',
    },
    {
        label: 'Macro F1',
        field: 'macroF1',
        tooltip: 'Average of F1 score across all classes.',
        format: 'percentage',
    },
    {
        label: 'Weighted Precision',
        field: 'weightedPrecision',
        tooltip:
            'Average of precision across all classes, weighted by the number of samples in each class.',
        format: 'percentage',
    },
    {
        label: 'Weighted Recall',
        field: 'weightedRecall',
        tooltip:
            'Average recall across all classes, weighted by the number of samples in each class.',
        format: 'percentage',
    },
    {
        label: 'Weighted F1',
        field: 'weightedF1',
        tooltip:
            'Average F1 score across all classes, weighted by the number of samples in each class.',
        format: 'percentage',
    },
];

export function MetricsDisplay({ metrics }: MetricsDisplayProps) {
    if (metrics.type === 'binary') {
        return (
            <div className="p-4 rounded-lg bg-primary-foreground">
                <div className="grid grid-cols-3 md:grid-cols-3 gap-3 text-sm">
                    {PER_CLASS_METRICS_CONFIG.map((metricConfig) => (
                        <MetricCard
                            key={metricConfig.label}
                            label={metricConfig.label}
                            value={metricConfig.field ? metrics[metricConfig.field] : null}
                            tooltip={metricConfig.tooltip}
                            format={metricConfig.format}
                        />
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div className="p-4 rounded-lg bg-primary-foreground">
            <div className="grid grid-cols-3 md:grid-cols-3 gap-3 text-sm">
                {FULL_METRICS_CONFIG.map((metricConfig) => (
                    <MetricCard
                        key={metricConfig.label}
                        label={metricConfig.label}
                        value={metricConfig.field ? metrics[metricConfig.field] : null}
                        tooltip={metricConfig.tooltip}
                        format={metricConfig.format}
                    />
                ))}
            </div>
        </div>
    );
}
