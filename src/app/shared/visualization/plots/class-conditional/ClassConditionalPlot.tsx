import { useState } from 'react';
import Plot from 'react-plotly.js';
import type { PlotProps } from '@/app/shared/registry/types';
import type { NaiveBayesTrainingReport } from '@/app/models/naive-bayes/types';
import { useClassConditionalPlotData } from './hooks/useClassConditionalPlotData';
import { Select } from '@/app/shared/ui/basic/select';

/**
 * ClassConditionalPlot component
 * @param dataset - The dataset
 * @param report - The report
 * @returns The ClassConditionalPlot component
 */
export function ClassConditionalPlot({ dataset, report }: PlotProps<NaiveBayesTrainingReport>) {
    const { params } = report;
    const { headers, categories } = dataset;

    // Get feature headers (excluding the first column which is typically the target)
    const featureHeaders = headers.slice(1);
    const [selectedFeatureIndex, setSelectedFeatureIndex] = useState(0);

    const plotData = useClassConditionalPlotData(
        params,
        categories || [],
        headers,
        selectedFeatureIndex,
    );

    const { traces = [], featureName } = plotData ?? {};

    return (
        <div className="w-full py-4 bg-background">
            <div className="flex flex-row justify-end">
                <Select
                    value={selectedFeatureIndex.toString()}
                    onValueChange={(value: string) => setSelectedFeatureIndex(parseInt(value))}
                >
                    <Select.Trigger id="feature-select" size="xs" className="border-0 shadow-none">
                        <Select.Value placeholder="Select feature" />
                    </Select.Trigger>
                    <Select.Content>
                        {featureHeaders.map((header, index) => (
                            <Select.Item key={index} value={index.toString()}>
                                {header}
                            </Select.Item>
                        ))}
                    </Select.Content>
                </Select>
            </div>
            <div className="w-full h-100 bg-background">
                <Plot
                    data={traces}
                    layout={{
                        xaxis: {
                            title: { text: featureName },
                            showgrid: true,
                        },
                        yaxis: {
                            title: { text: 'Probability Density' },
                            showgrid: true,
                        },
                        legend: {
                            x: 0.5,
                            y: -0.2,
                            xanchor: 'center',
                            yanchor: 'top',
                            orientation: 'h',
                        },
                        margin: { l: 60, r: 40, t: 40, b: 80 },
                        hovermode: 'closest',
                    }}
                    style={{ width: '100%', height: '100%' }}
                    config={{ displayModeBar: false, staticPlot: false, responsive: true }}
                    useResizeHandler
                />
            </div>
        </div>
    );
}
