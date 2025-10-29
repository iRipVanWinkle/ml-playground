import { useEffect, useState } from 'react';
import { Loader } from 'lucide-react';

type DelayedLoaderProps = {
    flag: boolean;
    children: React.ReactNode;
};

export function DelayedLoader({ flag, children }: DelayedLoaderProps) {
    const [showLoader, setShowLoader] = useState(false);
    useEffect(() => {
        let timer: NodeJS.Timeout | undefined;
        if (flag) {
            timer = setTimeout(() => setShowLoader(true), 100);
        } else {
            timer = setTimeout(() => setShowLoader(false), 0);
        }
        return () => clearTimeout(timer);
    }, [flag]);
    return <>{flag && showLoader ? <Loader className="animate-spin" /> : children}</>;
}
