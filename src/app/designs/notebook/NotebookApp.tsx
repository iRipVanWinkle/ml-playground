import { TaskSwitcher } from '@/app/features/switch-task';
import { DataSection } from '@/app/widgets/data-section';
import { ModelSection } from '@/app/widgets/model-section';
import { SystemSettings } from '@/app/widgets/settings-section';
import { TrainingSection } from '@/app/widgets/training-section';

/**
 * Notebook design — scaffold.
 *
 * A deliberately distinct, single-column "notebook" arrangement (training-first,
 * stacked cells) that currently reuses the shared widgets. As the notebook design
 * matures, replace these with notebook-specific widgets/features under
 * `designs/notebook/widgets` and `designs/notebook/features`, resolving model
 * views via `composedRegistry.getView(modelType, 'notebook')`.
 */
export function NotebookApp() {
    return (
        <main
            className="flex flex-col gap-6 px-4 py-6 m-auto w-full max-w-5xl"
            data-design="notebook"
        >
            <TaskSwitcher />

            <section className="flex flex-col gap-6">
                <TrainingSection />

                <div className="grid gap-6 grid-cols-1 md:grid-cols-2">
                    <DataSection />

                    <ModelSection />
                </div>

                <SystemSettings />
            </section>
        </main>
    );
}

export default NotebookApp;
