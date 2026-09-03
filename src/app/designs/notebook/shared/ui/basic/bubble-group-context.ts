import * as React from 'react';

type BubbleGroupContextValue = {
    isSelected: (value: string) => boolean;
    toggle: (value: string) => void;
};

const BubbleGroupContext = React.createContext<BubbleGroupContextValue | null>(null);

function useOptionalBubbleGroupContext() {
    return React.useContext(BubbleGroupContext);
}

export { BubbleGroupContext, useOptionalBubbleGroupContext, type BubbleGroupContextValue };
