import { useCallback } from 'react';
import type { TaskType } from './shared/types';
import { Toaster } from './shared/ui';
import { getModelRegistry } from './models/ui-registry';
import { ThemeProvider } from './features/change-theme';
import { TaskSwitcher } from './features/switch-task';
import { useResetTrainingReport } from './features/visualize-training';
import { useSetModelType } from './features/configure-model';
import { Footer } from './widgets/footer';
import { Header } from './widgets/header';
import { DataSection } from './widgets/data-section';
import { ModelSection } from './widgets/model-section';
import { SystemSettings } from './widgets/settings-section';
import { TrainingSection } from './widgets/training-section';
import { useIsTraining, useResetTrainingControls } from './features/control-training';

import './App.css';
import { useResetDataset } from './features/load-dataset';

const modelRegistry = getModelRegistry();

function App() {
    const isTraining = useIsTraining();

    const setModelType = useSetModelType();
    const resetControls = useResetTrainingControls();
    const resetReport = useResetTrainingReport();
    const resetDataset = useResetDataset();

    const handleTaskChange = useCallback(
        (taskType: TaskType) => {
            const models = modelRegistry.getForTask(taskType);
            const modelType = models[0].key;

            setModelType(modelType, taskType);
            resetReport(modelType, taskType);
            resetControls();
            resetDataset();
        },
        [resetDataset, setModelType, resetReport, resetControls],
    );

    const handleDatasetChange = useCallback(() => {
        resetReport();
        resetControls();
    }, [resetReport, resetControls]);

    return (
        <ThemeProvider>
            <div className="App">
                <Header />
                <main className="grid gap-3">
                    <TaskSwitcher disabled={isTraining} onChange={handleTaskChange} />

                    <div className="grid gap-6 grid-cols-1 lg:grid-cols-3">
                        <div className="lg:col-span-1 flex flex-col gap-6">
                            <DataSection onChange={handleDatasetChange} />

                            <ModelSection />

                            <SystemSettings />
                        </div>

                        <div className="lg:col-span-2">
                            <TrainingSection />
                        </div>
                    </div>
                </main>

                <Footer />
            </div>
            <Toaster position="top-right" expand closeButton richColors />
        </ThemeProvider>
    );
}

export default App;
