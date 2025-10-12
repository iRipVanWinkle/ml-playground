import LinearPlots from './LinearPlots';
import LogisticPlots from './LogisticPlots';
import { useModelSettings, useTaskType, useTrainingReport } from '@/app/store';
import { useData } from '@/app/features/load-dataset';
import { Controls } from './Controls';
import { LossHistoryPlot } from './LossHistoryPlot';
import { arrayAvg } from './helpers/arrayAvg';
import { Card, Progress } from '@/app/shared/ui';

export default function TrainingSection() {
    const modelSettings = useModelSettings();
    const data = useData();
    const { trainInputFeatures, categories } = useData();
    const taskType = useTaskType();
    const report = useTrainingReport();

    const isRegression = taskType === 'regression';
    const isClassification = taskType === 'classification';

    let maxIterations = 0;
    if (modelSettings.type !== 'tree') {
        maxIterations = modelSettings.optimizer.maxIterations;
    }

    const { trainAccuracy, testAccuracy, testLoss, trainLossHistory } = report;
    const trainLoss = trainLossHistory.at(-1) ?? [];
    const avgTrainLoss = isRegression ? trainLoss.at(-1) : arrayAvg(trainLoss);

    const currentIteration = arrayAvg(report.iterations);

    return (
        <Card>
            <Card.Content className="grid gap-4">
                <div className="grid gap-4 sticky top-0 z-10 bg-card/80 backdrop-blur-md pt-4 -mt-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div className="flex gap-4">
                            <Controls />
                        </div>

                        <div
                            className="flex items-center justify-end"
                            data-testid="training-progress"
                        >
                            {currentIteration.toFixed(0)}/{maxIterations}
                        </div>
                    </div>

                    <Progress
                        value={currentIteration}
                        max={maxIterations}
                        className="w-full bg-gray-200 rounded-full h-0.25"
                    />
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    {isClassification && (
                        <>
                            <div>
                                Train Accuracy:{' '}
                                <div className="font-bold">
                                    {trainAccuracy ? (trainAccuracy * 100).toFixed(2) + '%' : '--'}
                                </div>
                            </div>
                            <div>
                                Test Accuracy:{' '}
                                <div className="font-bold">
                                    {testAccuracy ? (testAccuracy * 100).toFixed(2) + '%' : '--'}
                                </div>
                            </div>
                        </>
                    )}
                    {isRegression && (
                        <>
                            <div>
                                Train Loss:{' '}
                                <div className="font-bold">
                                    {avgTrainLoss ? avgTrainLoss.toFixed(4) : '--'}
                                </div>
                            </div>
                            <div>
                                Test Loss:{' '}
                                <div className="font-bold">
                                    {testLoss ? testLoss.toFixed(4) : '--'}
                                </div>
                            </div>
                        </>
                    )}
                </div>
                <div className="flex flex-col gap-4">
                    <div className="min-h-120 bg-muted rounded-lg flex items-center justify-center posotion-relative">
                        {isRegression && <LinearPlots data={data} report={report} />}
                        {isClassification && <LogisticPlots data={data} results={report} />}
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
            </Card.Content>
        </Card>
    );
}
