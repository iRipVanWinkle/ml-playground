import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { TrainingController } from './TrainingController';
import { EVENT_LOOP_YIELD_MS, PAUSE_CHECK_INTERVAL_MS } from '../constants';
import type { TrainingEventEmitter } from '../types';

describe('TrainingController', () => {
    let mockEventEmitter: TrainingEventEmitter;
    let controller: TrainingController;

    beforeEach(() => {
        mockEventEmitter = {
            emit: vi.fn(),
            on: vi.fn(),
            off: vi.fn(),
        } as unknown as TrainingEventEmitter;
        controller = new TrainingController(mockEventEmitter);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    describe('basic state transitions', () => {
        it('should initialize with correct default state', () => {
            const defaultController = new TrainingController();
            expect(defaultController.isTrainingStopped).toBe(false);
        });

        it('should handle stop()', () => {
            controller.stop();
            expect(controller.isTrainingStopped).toBe(true);
            expect(mockEventEmitter.emit).toHaveBeenCalledWith('state', 'stopped');
        });

        it('should handle pause()', () => {
            controller.pause();
            expect(mockEventEmitter.emit).toHaveBeenCalledWith('state', 'paused');
            expect(controller.isTrainingStopped).toBe(false);
        });

        it('should handle resume()', () => {
            controller.pause();
            controller.resume();
            expect(mockEventEmitter.emit).toHaveBeenCalledWith('state', 'training');
            expect(controller.isTrainingStopped).toBe(false);
        });

        it('should handle step()', () => {
            controller.step();
            expect(mockEventEmitter.emit).toHaveBeenCalledWith('state', 'stepped-forward');
        });
    });

    describe('handleControlFlow', () => {
        beforeEach(() => {
            vi.useFakeTimers();
        });

        afterEach(() => {
            vi.useRealTimers();
        });

        it('should yield to event loop if sync backend', async () => {
            const promise = controller.handleControlFlow(true);

            await vi.advanceTimersByTimeAsync(EVENT_LOOP_YIELD_MS);

            await promise;
        });

        it('should not yield to event loop if async backend', async () => {
            const promise = controller.handleControlFlow(false);

            await promise;
        });

        it('should block when paused and unblock when resumed', async () => {
            controller.pause();

            let resolved = false;
            const promise = controller.handleControlFlow(false).then(() => {
                resolved = true;
            });

            await vi.advanceTimersByTimeAsync(PAUSE_CHECK_INTERVAL_MS / 2);
            expect(resolved).toBe(false);

            // Still paused
            await vi.advanceTimersByTimeAsync(PAUSE_CHECK_INTERVAL_MS);
            expect(resolved).toBe(false);

            // Resume and allow the next check to pass
            controller.resume();
            await vi.advanceTimersByTimeAsync(PAUSE_CHECK_INTERVAL_MS);

            await promise;
            expect(resolved).toBe(true);
        });

        it('should unblock when stopped', async () => {
            controller.pause();

            let resolved = false;
            const promise = controller.handleControlFlow(false).then(() => {
                resolved = true;
            });

            await vi.advanceTimersByTimeAsync(PAUSE_CHECK_INTERVAL_MS / 2);
            expect(resolved).toBe(false);

            controller.stop();
            await vi.advanceTimersByTimeAsync(PAUSE_CHECK_INTERVAL_MS);

            await promise;
            expect(resolved).toBe(true);
        });

        it('should unblock when stepped and reset stepRequested', async () => {
            controller.pause();

            let resolved = false;
            const promise = controller.handleControlFlow(false).then(() => {
                resolved = true;
            });

            await vi.advanceTimersByTimeAsync(PAUSE_CHECK_INTERVAL_MS / 2);
            expect(resolved).toBe(false);

            controller.step();
            await vi.advanceTimersByTimeAsync(PAUSE_CHECK_INTERVAL_MS);

            await promise;
            expect(resolved).toBe(true);

            // The controller should pause again if handleControlFlow is called because stepRequested was reset
            let resolved2 = false;
            const promise2 = controller.handleControlFlow(false).then(() => {
                resolved2 = true;
            });

            await vi.advanceTimersByTimeAsync(PAUSE_CHECK_INTERVAL_MS);
            expect(resolved2).toBe(false);

            // cleanup
            controller.resume();
            await vi.advanceTimersByTimeAsync(PAUSE_CHECK_INTERVAL_MS);
            await promise2;
        });
    });
});
