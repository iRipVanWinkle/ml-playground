import { describe, it, expect, vi, beforeEach } from 'vitest';
import { EventEmitter } from './EventEmitter';

describe('EventEmitter', () => {
    let emitter: EventEmitter;

    beforeEach(() => {
        emitter = new EventEmitter();
    });

    describe('on()', () => {
        it('should register a callback for an event', async () => {
            const callback = vi.fn();
            emitter.on('test-event', callback);

            await emitter.emit('test-event');

            expect(callback).toHaveBeenCalledTimes(1);
        });

        it('should register multiple callbacks for the same event', async () => {
            const callback1 = vi.fn();
            const callback2 = vi.fn();

            emitter.on('test-event', callback1);
            emitter.on('test-event', callback2);

            await emitter.emit('test-event');

            expect(callback1).toHaveBeenCalledTimes(1);
            expect(callback2).toHaveBeenCalledTimes(1);
        });
    });

    describe('emit()', () => {
        it('should call callbacks with provided arguments', async () => {
            const callback = vi.fn();
            emitter.on('test-event', callback);

            await emitter.emit('test-event', 'arg1', 42, { key: 'value' });

            expect(callback).toHaveBeenCalledWith('arg1', 42, { key: 'value' });
        });

        it('should not throw when emitting an event with no registered callbacks', async () => {
            await expect(emitter.emit('unregistered-event')).resolves.not.toThrow();
        });

        it('should handle async callbacks correctly', async () => {
            const order: number[] = [];
            const callback1 = vi.fn(async () => {
                await new Promise((resolve) => setTimeout(resolve, 10));
                order.push(1);
            });
            const callback2 = vi.fn(async () => {
                order.push(2);
            });

            emitter.on('async-event', callback1);
            emitter.on('async-event', callback2);

            await emitter.emit('async-event');

            expect(callback1).toHaveBeenCalledTimes(1);
            expect(callback2).toHaveBeenCalledTimes(1);
            // Since emit uses Promise.all, order of completion depends on the callback implementation.
            // Both should be completed after emit resolves.
            expect(order).toHaveLength(2);
            expect(order).toContain(1);
            expect(order).toContain(2);
        });
    });

    describe('off()', () => {
        it('should remove a specific callback for an event', async () => {
            const callback1 = vi.fn();
            const callback2 = vi.fn();

            emitter.on('test-event', callback1);
            emitter.on('test-event', callback2);

            emitter.off('test-event', callback1);

            await emitter.emit('test-event');

            expect(callback1).not.toHaveBeenCalled();
            expect(callback2).toHaveBeenCalledTimes(1);
        });

        it('should remove all callbacks for an event if no callback is provided', async () => {
            const callback1 = vi.fn();
            const callback2 = vi.fn();

            emitter.on('test-event', callback1);
            emitter.on('test-event', callback2);

            emitter.off('test-event');

            await emitter.emit('test-event');

            expect(callback1).not.toHaveBeenCalled();
            expect(callback2).not.toHaveBeenCalled();
        });

        it('should not throw when trying to remove a callback from an unregistered event', () => {
            expect(() => {
                emitter.off('unregistered-event');
            }).not.toThrow();
        });
    });

    describe('clear()', () => {
        it('should remove all registered events and their callbacks', async () => {
            const callback1 = vi.fn();
            const callback2 = vi.fn();

            emitter.on('event1', callback1);
            emitter.on('event2', callback2);

            emitter.clear();

            await emitter.emit('event1');
            await emitter.emit('event2');

            expect(callback1).not.toHaveBeenCalled();
            expect(callback2).not.toHaveBeenCalled();
        });
    });
});
