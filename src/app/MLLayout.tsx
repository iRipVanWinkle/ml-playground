import DataSection from './sections/data/DataSection';
import ModelSection from './sections/ModelSection';
import ResultSection from './sections/training/TrainingSection';
import { Tabs, TabsList, TabsTrigger } from './components/ui/tabs';
import { setTaskType, useIsTraining, useTaskType, type TaskType } from './store';

export default function MLLayout() {
    const isTraining = useIsTraining();
    const taskType = useTaskType();

    const handleTaskTypeChange = (taskType: string) => {
        setTaskType(taskType as TaskType);
    };

    return (
        <div className="grid gap-6 grid-cols-1 lg:grid-cols-3">
            <div className="lg:col-span-1 flex flex-col gap-6">
                <Tabs
                    defaultValue={taskType}
                    className="w-full"
                    onValueChange={handleTaskTypeChange}
                >
                    <TabsList>
                        <TabsTrigger value="regression" disabled={isTraining}>
                            Regression
                        </TabsTrigger>
                    </TabsList>
                </Tabs>

                {/* Data Section */}
                <DataSection />

                {/* Model Section */}
                <ModelSection />
            </div>

            <div className="lg:col-span-2">
                {/* Results Section */}
                <ResultSection />
            </div>
        </div>
    );
}
