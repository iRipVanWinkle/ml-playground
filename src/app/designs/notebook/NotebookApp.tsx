import { TaskSection } from './features/task-section';
import { DatasetSection } from './features/dataset-section';
import { TransformSection } from './features/transform-section';
import { ModelSection } from './features/model-section';
import { TrainingSection } from './features/training-section';
import { VisualizeSection } from './features/visualize-section';
import { UserExample } from './features/user-example';

export function NotebookApp() {
    return (
        <main
            className="flex flex-col gap-6 px-4 py-6 m-auto w-full max-w-5xl"
            data-design="notebook"
        >
            <TaskSection />

            <DatasetSection />

            <TransformSection />

            <ModelSection />

            <TrainingSection />

            <VisualizeSection />

            <UserExample />
        </main>
    );
}

export default NotebookApp;
