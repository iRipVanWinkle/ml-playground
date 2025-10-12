export const NORMALIZATION_METHODS = [
    {
        value: 'none',
        label: 'None',
    },
    {
        value: 'zscore',
        label: 'Z-Score',
    },
    {
        value: 'linear',
        label: 'Min-Max',
    },
    {
        value: 'log',
        label: 'Log',
    },
] as const;
