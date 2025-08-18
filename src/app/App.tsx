import MLLayout from './MLLayout';
import { Toaster } from './components/ui/sonner';

import './App.css';

function App() {
    return (
        <>
            <MLLayout />
            <Toaster position="top-center" expand richColors />
        </>
    );
}

export default App;
