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
        tooltip: 'Correlation coefficient between observed and predicted classifications.',
        format: 'decimal',
    },
    {
        label: "Cohen's Kappa",
        field: 'cohensKappa',
        tooltip: 'Agreement between predictions and actuals, accounting for chance.',
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
        tooltip: 'Harmonic mean of precision and recall.',
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
        tooltip: 'Correlation coefficient between observed and predicted classifications.',
        format: 'decimal',
    },
    {
        label: "Cohen's Kappa",
        field: 'cohensKappa',
        tooltip: 'Agreement between predictions and actuals, accounting for chance.',
        format: 'decimal',
    },
    {
        label: 'Macro Precision',
        field: 'macroPrecision',
        tooltip: 'Unweighted mean of precision across all classes.',
        format: 'percentage',
    },
    {
        label: 'Macro Recall',
        field: 'macroRecall',
        tooltip: 'Unweighted mean of recall across all classes.',
        format: 'percentage',
    },
    {
        label: 'Macro F1',
        field: 'macroF1',
        tooltip: 'Unweighted mean of F1 score across all classes.',
        format: 'percentage',
    },
    {
        label: 'Weighted Precision',
        field: 'weightedPrecision',
        tooltip: 'Precision averaged across classes, weighted by class support.',
        format: 'percentage',
    },
    {
        label: 'Weighted Recall',
        field: 'weightedRecall',
        tooltip: 'Recall averaged across classes, weighted by class support.',
        format: 'percentage',
    },
    {
        label: 'Weighted F1',
        field: 'weightedF1',
        tooltip: 'F1 score averaged across classes, weighted by class support.',
        format: 'percentage',
    },
];

export function MetricsDisplay({ metrics }: MetricsDisplayProps) {
    if (metrics.type === 'binary') {
        return (
            <div className="p-3 rounded-lg bg-primary-foreground">
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
        <div className="p-3 rounded-lg bg-primary-foreground">
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
