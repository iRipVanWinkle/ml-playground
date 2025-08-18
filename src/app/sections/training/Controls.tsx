import { Button } from '@/app/components/ui/button';
import { useModel } from '@/app/hooks/useModel';
import { useHasData, useTrainingState } from '@/app/store';
import { Pause, Play, Square, StepForward } from 'lucide-react';

export function Controls() {
    const state = useTrainingState();
    const hasData = useHasData();
    const model = useModel();

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

    if (state === 'training') {
        buttons = (
            <>
                <Button onClick={handleStop}>
                    <Square />
                    Stop
                </Button>
                <Button onClick={handlePause}>
                    <Pause />
                    Pause
                </Button>
            </>
        );
    }

    if (state === 'paused') {
        buttons = (
            <>
                <Button onClick={handleStop}>
                    <Square />
                    Stop
                </Button>
                <Button onClick={handleResume}>
                    <Play />
                    Resume
                </Button>
                <Button onClick={handleStep}>
                    <StepForward />
                    Step
                </Button>
            </>
        );
    }

    return buttons;
}
