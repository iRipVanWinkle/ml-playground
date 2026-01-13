import { Loader, Pause, Play, Square, StepForward } from 'lucide-react';
import { StartButton } from './StartButton';
import { Button, DelayedLoader } from '@/app/shared/ui';
import { useModel } from '../hooks/useModel';
import { useTrainingState, usePendingAction } from '../store';
import type { TrainingSettings } from '@/app/models/types';

type ControlsProps = {
    hasData: boolean;
    snapshotTrainingSettings: () => TrainingSettings;
};

export function Controls({ hasData, snapshotTrainingSettings }: ControlsProps) {
    const state = useTrainingState();
    const pendingAction = usePendingAction();
    const model = useModel({ snapshotTrainingSettings });
    const isPendingStop = pendingAction === 'stop';
    const isPendingPause = pendingAction === 'pause';
    const isPendingResume = pendingAction === 'resume';
    const isPendingStep = pendingAction === 'step';

    const handleTrain = (byStep = false) => {
        model.train({ byStep });
    };
    const handleStop = () => model.stop();
    const handlePause = () => model.pause();
    const handleResume = () => model.resume();
    const handleStep = () => model.step();

    let buttons = <StartButton onTrain={handleTrain} disabled={!hasData} />;

    if (state === 'preparing') {
        buttons = (
            <Button disabled>
                <Loader className="animate-spin" />
                Dataset Preparing...
            </Button>
        );
    }

    if (state === 'training') {
        buttons = (
            <>
                <Button onClick={handleStop} disabled={isPendingStop} data-testid="stop-training">
                    <DelayedLoader flag={isPendingStop}>
                        <Square />
                    </DelayedLoader>
                    Stop
                </Button>
                <Button
                    onClick={handlePause}
                    disabled={isPendingPause || isPendingStop}
                    data-testid="pause-training"
                >
                    <DelayedLoader flag={isPendingPause}>
                        <Pause />
                    </DelayedLoader>
                    Pause
                </Button>
            </>
        );
    }

    if (state === 'paused') {
        buttons = (
            <>
                <Button onClick={handleStop} disabled={isPendingStop} data-testid="stop-training">
                    <DelayedLoader flag={isPendingStop}>
                        <Square />
                    </DelayedLoader>
                    Stop
                </Button>
                <Button
                    onClick={handleResume}
                    disabled={isPendingResume || isPendingStop}
                    data-testid="resume-training"
                >
                    <DelayedLoader flag={isPendingResume}>
                        <Play />
                    </DelayedLoader>
                    Resume
                </Button>
                <Button
                    onClick={handleStep}
                    disabled={isPendingStep || isPendingStop}
                    data-testid="step-forward"
                >
                    <DelayedLoader flag={isPendingStep}>
                        <StepForward />
                    </DelayedLoader>
                    Step
                </Button>
            </>
        );
    }

    return buttons;
}
