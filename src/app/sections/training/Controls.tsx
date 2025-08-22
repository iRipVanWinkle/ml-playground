import { Button } from '@/app/components/ui/button';
import { DelayedLoader } from '@/app/components/ui/delayed-loader';
import { useModel } from '@/app/hooks/useModel';
import { useHasData, usePendingAction, useTrainingState } from '@/app/store';
import { Loader, Pause, Play, Square, StepForward } from 'lucide-react';

export function Controls() {
    const state = useTrainingState();
    const pendingAction = usePendingAction();
    const hasData = useHasData();
    const model = useModel();

    const isPendingStop = pendingAction === 'stop';
    const isPendingPause = pendingAction === 'pause';
    const isPendingResume = pendingAction === 'resume';
    const isPendingStep = pendingAction === 'step';

    const handleTrain = () => model.train();
    const handleStop = () => model.stop();
    const handlePause = () => model.pause();
    const handleResume = () => model.resume();
    const handleStep = () => model.step();

    let buttons = (
        <Button onClick={handleTrain} disabled={!hasData}>
            <Play />
            Start Training
        </Button>
    );

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
                <Button onClick={handleStop} disabled={isPendingStop}>
                    <DelayedLoader flag={isPendingStop}>
                        <Square />
                    </DelayedLoader>
                    Stop
                </Button>
                <Button onClick={handlePause} disabled={isPendingPause || isPendingStop}>
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
                <Button onClick={handleStop} disabled={isPendingStop}>
                    <DelayedLoader flag={isPendingStop}>
                        <Square />
                    </DelayedLoader>
                    Stop
                </Button>
                <Button onClick={handleResume} disabled={isPendingResume || isPendingStop}>
                    <DelayedLoader flag={isPendingResume}>
                        <Play />
                    </DelayedLoader>
                    Resume
                </Button>
                <Button onClick={handleStep} disabled={isPendingStep || isPendingStop}>
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
