import { Toaster } from './shared/ui';
import { ThemeProvider } from './features/change-theme';
import { TaskSwitcher } from './features/switch-task';
import { Footer } from './widgets/footer';
import { Header } from './widgets/header';
import { DataSection } from './widgets/data-section';
import { ModelSection } from './widgets/model-section';
import { SystemSettings } from './widgets/settings-section';
import { TrainingSection } from './widgets/training-section';

import './App.css';

function App() {
    return (
        <ThemeProvider>
            <div className="App">
                <Header />
                <main className="grid gap-3">
                    <TaskSwitcher />

                    <div className="grid gap-6 grid-cols-1 lg:grid-cols-3">
                        <div className="lg:col-span-1 flex flex-col gap-6">
                            <DataSection />

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
