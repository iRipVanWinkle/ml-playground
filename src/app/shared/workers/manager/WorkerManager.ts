type PendingRequest<TResponse> = {
    resolve: (response: TResponse) => void;
    reject: (error: Error) => void;
    timeout?: ReturnType<typeof setTimeout>;
};

type ResponseMessage<TResponse> = {
    type: string;
    payload: TResponse;
    requestId?: string;
    sentAt?: number;
};

/* eslint-disable @typescript-eslint/no-explicit-any */
type MessageCallback = (...args: any[]) => void;

export class WorkerManager<TMessage, TResponse> {
    private workerFactory: () => Worker;
    private worker: Worker | null = null;
    private messageHandlers = new Map<string, Set<(payload: unknown) => void>>();
    private pendingRequests: Map<string, PendingRequest<TResponse>> = new Map();

    constructor(workerFactory: () => Worker) {
        this.workerFactory = workerFactory;
    }

    on(type: string, handler: MessageCallback): () => void {
        if (!this.messageHandlers.has(type)) {
            this.messageHandlers.set(type, new Set());
        }

        this.messageHandlers.get(type)!.add(handler);

        return () => {
            this.messageHandlers.get(type)?.delete(handler);
        };
    }

    postMessage(
        message: TMessage,
        options?: {
            transfer?: Transferable[];
        },
    ): void {
        const requestId = crypto.randomUUID();
        const messageWithId = {
            ...message,
            requestId,
            sentAt: import.meta.env.DEV ? performance.now() + performance.timeOrigin : undefined,
        };

        if (options?.transfer) {
            this.getWorker().postMessage(messageWithId, options.transfer);
        } else {
            this.getWorker().postMessage(messageWithId);
        }
    }

    postMessageAsync(
        message: TMessage,
        options?: {
            transfer?: Transferable[];
            timeout?: number;
        },
    ): Promise<TResponse> {
        const requestId = crypto.randomUUID();
        return new Promise((resolve, reject) => {
            const timeout = options?.timeout;
            const timeoutHandle = timeout
                ? setTimeout(() => {
                      this.pendingRequests.delete(requestId);
                      reject(new Error(`Request timeout after ${timeout}ms`));
                  }, timeout)
                : undefined;

            this.pendingRequests.set(requestId, {
                resolve,
                reject,
                timeout: timeoutHandle,
            });
            const messageWithId = {
                ...message,
                requestId,
                sentAt: import.meta.env.DEV
                    ? performance.now() + performance.timeOrigin
                    : undefined,
            };

            if (options?.transfer) {
                this.getWorker().postMessage(messageWithId, options.transfer);
            } else {
                this.getWorker().postMessage(messageWithId);
            }
        });
    }

    terminate(): void {
        for (const [, pending] of this.pendingRequests) {
            clearTimeout(pending.timeout);
            pending.reject(new Error('Worker terminated'));
        }
        this.pendingRequests.clear();

        const worker = this.getWorker();

        worker.removeEventListener('message', this.handleMessage);
        worker.removeEventListener('error', this.handleError);
        worker.removeEventListener('messageerror', this.handleMessageError);

        worker.terminate();

        this.messageHandlers.clear();

        this.worker = null;
    }

    private getWorker(): Worker {
        if (!this.worker) {
            this.worker = this.workerFactory();
            this.setupListeners(this.worker);
        }
        return this.worker;
    }

    private setupListeners(worker: Worker): void {
        worker.addEventListener('message', this.handleMessage);
        worker.addEventListener('error', this.handleError);
        worker.addEventListener('messageerror', this.handleMessageError);
    }

    private handleMessage = (event: MessageEvent<TResponse>): void => {
        const { type, payload, requestId, sentAt } = event.data as ResponseMessage<TResponse>;

        if (import.meta.env.DEV && sentAt) {
            const now = performance.now() + performance.timeOrigin;
            const latency = now - sentAt;
            console.log(
                `%c[Worker -> Client] %c${type} %clatency: ${latency.toFixed(2)}ms`,
                'color: #00bcd4; font-weight: bold',
                'color: inherit',
                'color: #4caf50',
            );
        }

        const pending = this.pendingRequests.get(requestId!);
        if (pending) {
            if (type === 'error') {
                pending.reject(new Error(payload as string));
            } else {
                pending.resolve(payload);
            }

            clearTimeout(pending.timeout);
            this.pendingRequests.delete(requestId!);
        }

        const handlers = this.messageHandlers.get(type);
        if (handlers) {
            handlers.forEach((handler) => handler(payload));
        }
    };

    private handleError = (error: ErrorEvent): void => {
        const wrappedError = new Error(error.message);
        const handlers = this.messageHandlers.get('error');
        if (handlers) {
            handlers.forEach((handler) => handler(wrappedError));
        }
    };

    private handleMessageError = (event: MessageEvent): void => {
        const wrappedError = new Error(`Message serialization error: ${event}`);
        const handlers = this.messageHandlers.get('error');
        if (handlers) {
            handlers.forEach((handler) => handler(wrappedError));
        }
    };
}
