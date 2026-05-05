import { lazy, Suspense, type ComponentType, type CSSProperties } from 'react';
import type Plotly from 'plotly.js';
import type { Config } from 'plotly.js';
import type { PlotParams } from 'react-plotly.js';
import { usePlotlyLayout } from './hooks';
import { deepMerge } from './utils';

export const PlotlyHeatmap = createLazyChart(() => [import('plotly.js/lib/heatmap')]);

export const PlotlyScatter = createLazyChart(() => [import('plotly.js/lib/scatter')]);

export const PlotlyScatter3D = createLazyChart(() => [import('plotly.js/lib/scatter3d')]);

export const PlotlyScatterContour = createLazyChart(() => [
    import('plotly.js/lib/scatter'),
    import('plotly.js/lib/contour'),
]);

/**
 * Base configuration and style for the themed Plotly charts
 */

type PlotlyModuleParam = Parameters<typeof Plotly.register>[0];
type PlotlyModule = Exclude<PlotlyModuleParam, PlotlyModuleParam[]>;
type TraceModuleLoader = () => Promise<PlotlyModule>[];

const BASE_CONFIG: Partial<Config> = {
    displayModeBar: false,
    staticPlot: false,
    responsive: true,
};

const BASE_STYLE: Partial<CSSProperties> = {
    width: '100%',
    height: '100%',
};

/**
 * Creates a lazy-loaded, themed Plotly chart component.
 */
function createLazyChart(loadTraceModules: TraceModuleLoader): ComponentType<PlotParams> {
    const LazyPlot = lazy(() =>
        Promise.all([
            import('react-plotly.js/factory'),
            import('plotly.js/lib/core'),
            ...loadTraceModules(),
        ]).then(([plotlyFactory, Plotly, ...modules]) => {
            Plotly.register(modules);
            return { default: plotlyFactory.default(Plotly) };
        }),
    );

    function ThemedChart(props: PlotParams) {
        const baseLayout = usePlotlyLayout();

        const layout = deepMerge(baseLayout, props.layout ?? {});
        const config = deepMerge(BASE_CONFIG, props.config ?? {});
        const style = { ...BASE_STYLE, ...(props.style ?? {}) };

        return (
            <Suspense fallback={<div>Loading visualization...</div>}>
                <LazyPlot
                    {...props}
                    layout={layout}
                    config={config}
                    style={style}
                    useResizeHandler
                />
            </Suspense>
        );
    }

    return ThemedChart;
}
