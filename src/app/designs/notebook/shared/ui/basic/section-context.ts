import * as React from 'react';

type SectionContextValue = {
    /** Position of this section in the flow, shared with `StepNum`. */
    step?: number;
    /** Total number of sections in the flow, shared with `StepNum`. */
    total?: number;
};

const SectionContext = React.createContext<SectionContextValue | null>(null);

function useOptionalSectionContext() {
    return React.useContext(SectionContext);
}

export { SectionContext, useOptionalSectionContext, type SectionContextValue };
