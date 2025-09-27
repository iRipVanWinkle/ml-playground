import DataSection from './sections/data/DataSection';
import { ModelSection } from './sections/model';
import ResultSection from './sections/training/TrainingSection';
import { Tabs, TabsList, TabsTrigger } from './components/ui/enhanced-tabs';
import { setTaskType, useIsTraining, useTaskType, type TaskType } from './store';
import { SystemSettings } from './sections/settings/SystemSettings';
import { Shapes, TrendingUp } from 'lucide-react';

const TASK_TYPES = [
    {
        value: 'regression',
        label: 'Regression',
        description: 'Predict continuous values like prices, temperatures, or sales',
        icon: <TrendingUp />,
    },
    {
        value: 'classification',
        label: 'Classification',
        description: 'Categorize data into distinct classes or groups',
        icon: <Shapes />,
    },
];

export default function MLLayout() {
    const isTraining = useIsTraining();
    const taskType = useTaskType();

    const handleTaskTypeChange = (taskType: string) => {
        setTaskType(taskType as TaskType);
    };

    return (
        <div className="grid gap-3">
            <Tabs defaultValue={taskType} className="w-full" onValueChange={handleTaskTypeChange}>
                <TabsList variant="underline">
                    {TASK_TYPES.map((tt) => (
                        <TabsTrigger
                            key={tt.value}
                            value={tt.value}
                            disabled={isTraining}
                            icon={tt.icon}
                        >
                            {tt.label}
                        </TabsTrigger>
                    ))}
                </TabsList>
            </Tabs>
            <div className="grid gap-6 grid-cols-1 lg:grid-cols-3">
                <div className="lg:col-span-1 flex flex-col gap-6">
                    {/* Data Section */}
                    <DataSection />

                    {/* Model Section */}
                    <ModelSection />

                    {/* System Settings Section */}
                    <SystemSettings />
                </div>

                <div className="lg:col-span-2">
                    {/* Results Section */}
                    <ResultSection />
                </div>
            </div>
        </div>
    );
}
