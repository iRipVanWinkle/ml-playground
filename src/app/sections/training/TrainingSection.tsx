import { Card, CardContent } from '@/app/components/ui/card';
import { Progress } from '@/app/components/ui/progress';
import { useData, useModelSettings, useTrainingReport } from '@/app/store';
import LinearPlots from './LinearPlots';
import { Controls } from './Controls';
import { LossHistoryPlot } from './LossHistoryPlot';
import { arrayAvg } from './arrayAvg';

export default function TrainingSection() {
    const modelSettings = useModelSettings();
    const data = useData();
    const { trainInputFeatures, categories } = useData();
    const report = useTrainingReport();

    const { maxIterations } = modelSettings.optimizer;

    const { testLoss, trainLossHistory } = report;
    const trainLoss = trainLossHistory.at(-1) ?? [];
    const avgTrainLoss = trainLoss.at(-1);

    const currentIteration = arrayAvg(report.iterations);

    return (
        <Card>
            <CardContent className="grid gap-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex gap-4">
                        <Controls />
                    </div>

                    <div className="flex items-center justify-end">
                        {currentIteration.toFixed(0)}/{maxIterations}
                    </div>
                </div>

                <Progress
                    value={currentIteration}
                    max={maxIterations}
                    className="w-full bg-gray-200 rounded-full h-0.25"
                />

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                        Train Loss:{' '}
                        <div className="font-bold">
                            {avgTrainLoss ? avgTrainLoss.toFixed(4) : '--'}
                        </div>
                    </div>
                    <div>
                        Test Loss:{' '}
                        <div className="font-bold">{testLoss ? testLoss.toFixed(4) : '--'}</div>
                    </div>
                </div>
                <div className="flex flex-col gap-4">
                    <div className="min-h-120 bg-muted rounded-lg flex items-center justify-center posotion-relative">
                        <LinearPlots data={data} report={report} />
                    </div>
                    <div className="h-80 bg-muted rounded-lg flex items-center justify-center">
                        {!!trainInputFeatures && (
                            <LossHistoryPlot
                                lossHistory={trainLossHistory}
                                categories={categories}
                            />
                        )}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
