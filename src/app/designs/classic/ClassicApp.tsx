import { TaskSwitcher } from '@/app/features/switch-task';
import { DataSection } from '@/app/widgets/data-section';
import { ModelSection } from '@/app/widgets/model-section';
import { SystemSettings } from '@/app/widgets/settings-section';
import { TrainingSection } from '@/app/widgets/training-section';

export function ClassicApp() {
    return (
        <main className="grid gap-6 px-2 py-6 m-auto max-w-7xl" data-design="classic">
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
    );
}

export default ClassicApp;
