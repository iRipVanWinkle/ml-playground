import type { ComponentType, ReactNode } from 'react';
import type { ModelSettings, ModelType } from '@/app/models/types';
import type { Dataset, TaskType, Transformation } from '@/app/shared/types';
import type { SettingsOf, TrainingReportOf } from './utils';

export type PlotVisualization<TKey extends ModelType> = {
    title: string;
    component: ComponentType<PlotProps<TrainingReportOf<TKey>>>;
};

type PlotsVisualization<TKey extends ModelType> =
    | Array<PlotVisualization<TKey>>
    | ((taskType: TaskType) => Array<PlotVisualization<TKey>>);

export interface ModelDefinition<TKey extends ModelType = ModelType> {
    key: TKey;
    label: string;
    taskTypes: TaskType[];

    defaultSettings: (taskType: TaskType) => SettingsOf<TKey>;
    settingsComponent: ComponentType<ModelSettingsComponentProps<SettingsOf<TKey>>>;

    defaultReport: (taskType: TaskType) => TrainingReportOf<TKey>;
    visualization: {
        metricsGridComponent: ComponentType<MainMetricsProps<TrainingReportOf<TKey>>>;
        modelDataPlotComponent: ComponentType<
            ModelDataPlotProps<TrainingReportOf<TKey>, SettingsOf<TKey>>
        >;
        plots?: PlotsVisualization<TKey>;
        parametersComponent?: ComponentType<ParametersVisualizationProps<TrainingReportOf<TKey>>>;
    };

    progress: {
        getProgressInfo: (params: {
            report: TrainingReportOf<TKey>;
            settings: SettingsOf<TKey>;
            dataset: Dataset;
        }) => ProgressInfo;
    };
}

export type ModelSettingsComponentProps<TSettings> = {
    taskType: TaskType;
    settings: TSettings;
    disabled: boolean;
    additionalParams?: {
        numCategories?: number;
    };
    onChange: (settings: Partial<TSettings>) => void;
};

export type MainMetricsProps<TTrainingReport> = {
    report: TTrainingReport;
};

export type ModelDataPlotProps<TTrainingReport, TSettings = ModelSettings> = {
    dataset: Dataset;
    report: TTrainingReport;
    modelSettings: TSettings;
};

export type PlotProps<TTrainingReport> = {
    dataset: Dataset;
    report: TTrainingReport;
};

export type ParametersVisualizationProps<TTrainingReport> = {
    dataset: Dataset;
    modelSettings: ModelSettings;
    transformations: Transformation[];
    report: TTrainingReport;
};

export type ProgressInfo =
    | {
          type: 'determinate';
          label: ReactNode;
          current: number;
          max: number;
      }
    | {
          type: 'indeterminate';
          label: ReactNode;
      };
