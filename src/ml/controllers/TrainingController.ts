import { EVENT_LOOP_YIELD_MS, PAUSE_CHECK_INTERVAL_MS } from '../constants';
import type { TrainingControl, TrainingEventEmitter } from '../types';

export class TrainingController implements TrainingControl {
    private isStopped = false;
    private isPaused = false;
    private stepRequested = false;

    private eventEmitter?: TrainingEventEmitter;

    constructor(eventEmitter?: TrainingEventEmitter) {
        this.eventEmitter = eventEmitter;
    }

    stop(): void {
        this.isPaused = false;
        this.isStopped = true;
        this.eventEmitter?.emit('state', 'stopped');
    }

    pause(): void {
        this.isPaused = true;
        this.eventEmitter?.emit('state', 'paused');
    }

    resume(): void {
        this.isPaused = false;
        this.eventEmitter?.emit('state', 'training');
    }

    step(): void {
        this.stepRequested = true;
        this.eventEmitter?.emit('state', 'stepped-forward');
    }

    /**
     * Indicates if the training process has been stopped.
     */
    get isTrainingStopped(): boolean {
        return this.isStopped;
    }

    /**
     * Handles control flow for the training process.
     * @param isSyncBackend - Indicates if the backend is synchronous (e.g., CPU). If true, yields control to the event loop.
     */
    async handleControlFlow(isSyncBackend = false): Promise<void> {
        if (isSyncBackend) {
            await new Promise((resolve) => setTimeout(resolve, EVENT_LOOP_YIELD_MS));
        }

        while (this.isPaused && !this.stepRequested && !this.isStopped) {
            await new Promise((resolve) => setTimeout(resolve, PAUSE_CHECK_INTERVAL_MS));
        }

        this.stepRequested = false;
    }
}
