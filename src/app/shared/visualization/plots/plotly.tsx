import { lazy, Suspense, type ComponentType } from 'react';
import type Plotly from 'plotly.js';
import type { PlotParams } from 'react-plotly.js';
import type createPlotlyComponent from 'react-plotly.js/factory';

type PlotlyModuleParam = Parameters<typeof Plotly.register>[0];
type PlotlyModule = Exclude<PlotlyModuleParam, PlotlyModuleParam[]>;
type LoaderType = () => Promise<
    [{ default: typeof createPlotlyComponent }, typeof Plotly, ...PlotlyModule[]]
>;

function createLazyPlotlyComponent(loader: LoaderType): ComponentType<PlotParams> {
    const LazyComponent = lazy(() =>
        loader().then(([factoryModule, Plotly, ...modules]) => {
            Plotly.register(modules);
            return { default: factoryModule.default(Plotly) };
        }),
    );

    function WrappedComponent(props: PlotParams) {
        return (
            <Suspense fallback={<div>Loading visualization...</div>}>
                <LazyComponent {...props} />
            </Suspense>
        );
    }

    return WrappedComponent;
}

export const PlotlyHeatmap = createLazyPlotlyComponent(() =>
    Promise.all([
        import('react-plotly.js/factory'),
        import('plotly.js/lib/core'),
        import('plotly.js/lib/heatmap'),
    ]),
);

export const PlotlyScatter = createLazyPlotlyComponent(() =>
    Promise.all([
        import('react-plotly.js/factory'),
        import('plotly.js/lib/core'),
        import('plotly.js/lib/scatter'),
    ]),
);

export const PlotlyScatter3D = createLazyPlotlyComponent(() =>
    Promise.all([
        import('react-plotly.js/factory'),
        import('plotly.js/lib/core'),
        import('plotly.js/lib/scatter3d'),
    ]),
);

export const PlotlyScatterContour = createLazyPlotlyComponent(() =>
    Promise.all([
        import('react-plotly.js/factory'),
        import('plotly.js/lib/core'),
        import('plotly.js/lib/scatter'),
        import('plotly.js/lib/contour'),
    ]),
);
