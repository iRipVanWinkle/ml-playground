import type { ComponentType, ReactNode } from 'react';
import type { ModelType } from '@/app/models/types';
import type { Dataset, TaskType } from '@/app/shared/types';
import type { SettingsOf, TrainingReportOf } from './utils';

export interface ModelDefinition<TKey extends ModelType = ModelType> {
    key: TKey;
    label: string;
    taskTypes: TaskType[];

    defaultSettings: (taskType?: TaskType) => SettingsOf<TKey>;
    settingsComponent: ComponentType<ModelSettingsComponentProps<SettingsOf<TKey>>>;

    visualization: {
        metricsGridComponent: ComponentType<MainMetricsProps<TrainingReportOf<TKey>>>;
        modelDataPlotComponent: ComponentType<ModelDataPlotProps<TrainingReportOf<TKey>>>;
        plots?: Array<{
            title: string;
            component: ComponentType<PlotProps<TrainingReportOf<TKey>>>;
        }>;
    };

    progress: {
        getProgressInfo: (
            report: TrainingReportOf<TKey>,
            settings: SettingsOf<TKey>,
        ) => ProgressInfo;
    };
}

export type ModelSettingsComponentProps<TSettings> = {
    taskType: TaskType;
    settings: TSettings;
    disabled: boolean;
    additionalParams?: {
        numCategories?: number;
    };
    onChange: (settings: TSettings) => void;
};

export type MainMetricsProps<TTrainingReport> = {
    report: TTrainingReport;
};

export type ModelDataPlotProps<TTrainingReport> = {
    dataset: Dataset;
    report: TTrainingReport;
};

export type PlotProps<TTrainingReport> = {
    dataset: Dataset;
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
