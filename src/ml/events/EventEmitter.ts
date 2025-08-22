/* eslint-disable @typescript-eslint/no-explicit-any */
type EventCallback = (...args: any[]) => void;

export class EventEmitter {
    private events: Map<string, EventCallback[]> = new Map();

    /**
     * Registers a callback for a specific event.
     * @param eventName - The name of the event.
     * @param callback - The callback function to register.
     */
    on(eventName: string, callback: EventCallback): void {
        if (!this.events.has(eventName)) {
            this.events.set(eventName, []);
        }
        this.events.get(eventName)!.push(callback);
    }

    /**
     * Removes a callback for a specific event.
     * If no callback is provided, all callbacks for the event are removed.
     * @param eventName - The name of the event.
     * @param callback - The callback function to remove (optional).
     */
    off(eventName: string, callback?: EventCallback): void {
        if (!this.events.has(eventName)) return;

        if (callback) {
            const callbacks = this.events.get(eventName)!;
            this.events.set(
                eventName,
                callbacks.filter((cb) => cb !== callback),
            );
        } else {
            this.events.delete(eventName);
        }
    }

    /**
     * Triggers an event and calls all registered callbacks with the provided arguments.
     * @param eventName - The name of the event.
     * @param args - Arguments to pass to the callback functions.
     */
    async emit(eventName: string, ...args: any[]): Promise<void> {
        if (!this.events.has(eventName)) return;

        const callbacks = this.events.get(eventName)!;
        await Promise.all(callbacks.map((callback) => callback(...args)));
    }

    /**
     * Clears all registered events and their callbacks.
     */
    clear(): void {
        this.events.clear();
    }
}
