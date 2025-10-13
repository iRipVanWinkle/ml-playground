export type TrainingState = 'init' | 'preparing' | 'training' | 'paused';
export type PendingAction = 'pause' | 'stop' | 'step' | 'resume' | null;

export type TrainingReport = {
    trainLossHistory: number[][];
    testLoss: number;
    trainAccuracy: number;
    testAccuracy: number;
    iterations: number[];
    predictionPredictedLabels?: number[][];
    trainPredictedLabels: number[][];
    testPredictedLabels: number[][];
    theta: number[][];
};

export type TrainingStore = {
    trainingState: TrainingState;
    pendingAction: PendingAction;
    trainingReport: TrainingReport;
};
