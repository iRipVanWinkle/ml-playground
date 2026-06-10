import { Suspense } from 'react';
import { DESIGN_SHELLS } from './designs/shells';
import { useDesign } from './features/switch-design';

export function DesignRoot() {
    const { design } = useDesign();
    const Shell = DESIGN_SHELLS[design];

    return (
        <Suspense
            fallback={
                <div className="grid min-h-screen place-items-center text-sm text-muted-foreground">
                    Loading…
                </div>
            }
        >
            <Shell />
        </Suspense>
    );
}
