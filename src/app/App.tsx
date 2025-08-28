import MLLayout from './MLLayout';
import { Toaster } from './components/ui/sonner';

import './App.css';

function App() {
    return (
        <>
            <MLLayout />
            <Toaster position="top-right" expand richColors closeButton />
        </>
    );
}

export default App;
