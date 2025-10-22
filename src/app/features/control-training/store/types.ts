export type TrainingState = 'init' | 'preparing' | 'training' | 'paused';
export type PendingAction = 'pause' | 'stop' | 'step' | 'resume' | null;

export type TrainingStore = {
    trainingState: TrainingState;
    pendingAction: PendingAction;
};
